"""Bounded multi-turn tool trajectories; only explicitly registered tools run."""
import json
import multiprocessing
import pickle
import re
from dataclasses import dataclass

import torch

from .generation import generate_tokens
from .rollout import RolloutBatch, TorchRolloutEngine, prompt_text


def parse_response(text):
    reasoning = None
    content = text
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        reasoning = match.group(1)
        content = text[:match.start()] + text[match.end():]
    elif "<think>" in text:
        content, reasoning = text.split("<think>", 1)
    if content.count("<tool_call>") != content.count("</tool_call>"):
        raise ValueError("incomplete tool call")
    calls = []
    for match in list(re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)):
        value = json.loads(match.group(1))
        value = value if isinstance(value, list) else [value]
        for call in value:
            function = call.get("function", call)
            name, arguments = function.get("name"), function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError("tool call requires a name and object arguments")
            calls.append({"id": call.get("id", "call_{}".format(len(calls))), "type": "function",
                          "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}})
    content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
    return {"content": content.strip(), "reasoning_content": reasoning, "tool_calls": calls}


def _tool_worker(connection, function, arguments):
    try:
        result = function(**arguments)
        connection.send({"ok": True, "result": result})
    except Exception as error:
        connection.send({"ok": False, "error": type(error).__name__ + ": " + str(error)})
    finally:
        connection.close()


@dataclass
class ToolDefinition:
    name: str
    function: object
    schema: dict
    timeout: float


class ToolRegistry:
    def __init__(self, max_output_chars=4096):
        self.tools = {}
        self.max_output_chars = max_output_chars

    def register(self, name, function, parameters=None, description="", timeout=5.0):
        if name in self.tools or timeout <= 0 or not callable(function):
            raise ValueError("invalid or duplicate tool")
        # Spawn gives a real timeout on Windows too. Tools must be importable
        # top-level functions, not lambdas/closures or unbounded thread jobs.
        try:
            pickle.dumps(function)
        except (TypeError, pickle.PicklingError, AttributeError) as error:
            raise TypeError("tools must be picklable top-level functions") from error
        schema = {"type": "function", "function": {"name": name, "description": description,
                                                   "parameters": parameters or {"type": "object", "properties": {}}}}
        self.tools[name] = ToolDefinition(name, function, schema, float(timeout))
        return self

    @property
    def schemas(self):
        return [tool.schema for tool in self.tools.values()]

    def execute(self, call):
        function = call["function"]
        name = function["name"]
        if name not in self.tools:
            return {"ok": False, "error": "unregistered tool: " + name}
        tool = self.tools[name]
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            return {"ok": False, "error": "arguments must be an object"}
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=_tool_worker, args=(child, tool.function, arguments), daemon=True)
        process.start()
        child.close()
        try:
            if parent.poll(tool.timeout):
                try:
                    output = parent.recv()
                except EOFError:
                    output = {"ok": False, "error": "tool worker exited without a result"}
            else:
                output = {"ok": False, "error": "tool timeout"}
        finally:
            if process.is_alive():
                process.terminate()
            process.join()
            parent.close()
        text = json.dumps(output, ensure_ascii=False, default=str)
        if len(text) > self.max_output_chars:
            return {"ok": False, "error": "tool output exceeded configured limit"}
        return output


class AgenticRolloutEngine(TorchRolloutEngine):
    def __init__(self, tokenizer, tools, max_turns=4, max_tool_calls=8, template=None):
        super().__init__(tokenizer, template)
        if max_turns < 1 or max_tool_calls < 1:
            raise ValueError("agent limits must be positive")
        self.tools, self.max_turns, self.max_tool_calls = tools, max_turns, max_tool_calls

    @torch.no_grad()
    def rollout(self, samples, num_generations=4, max_new_tokens=32, temperature=0.8, top_k=0, top_p=1.0):
        if self.policy is None:
            raise RuntimeError("synchronize the policy before rollout")
        device = next(self.policy.parameters()).device
        rows, label_rows, probability_rows, prompts, completions, trajectories = [], [], [], [], [], []
        for sample in samples:
            if isinstance(sample, dict):
                sample = dict(sample, tools=sample.get("tools", self.tools.schemas))
            prompt = prompt_text(sample, self.template)
            for _ in range(num_generations):
                tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                labels = [-100] * len(tokens)
                probabilities = [0.] * len(tokens)
                events, used_calls, completion = [], 0, ""
                for turn in range(self.max_turns):
                    capacity = getattr(self.policy.config, "max_position_embeddings", 32768) - len(tokens)
                    if capacity <= 0:
                        events.append({"stop": "context_limit"})
                        break
                    inputs = torch.tensor([tokens], dtype=torch.long, device=device)
                    result, sampled = generate_tokens(self.policy, inputs, max_new_tokens=min(max_new_tokens, capacity),
                                             temperature=temperature, top_k=top_k, top_p=top_p,
                                             eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
                                                      return_log_probs=True)
                    action = result[0, len(tokens):].tolist()
                    tokens.extend(action)
                    labels.extend(action)
                    probabilities.extend(sampled[0].tolist())
                    text_ids = action[:-1] if action and action[-1] == self.tokenizer.eos_token_id else action
                    completion = self.tokenizer.decode(text_ids, skip_special_tokens=False)
                    try:
                        parsed = parse_response(completion)
                    except (ValueError, TypeError):
                        events.append({"action": completion, "stop": "invalid_tool_call"})
                        break
                    event = {"action": completion, "tool_results": []}
                    events.append(event)
                    if not parsed["tool_calls"]:
                        event["stop"] = "answer"
                        break
                    for call in parsed["tool_calls"]:
                        if used_calls >= self.max_tool_calls:
                            event["stop"] = "tool_limit"
                            break
                        observation = self.tools.execute(call)
                        used_calls += 1
                        event["tool_results"].append({"call": call, "observation": observation})
                        observation_text = "\n<|im_start|>tool\n" + json.dumps(observation, ensure_ascii=False, default=str) + "<|im_end|>\n"
                        observation_ids = self.tokenizer.encode(observation_text, add_special_tokens=False)
                        tokens.extend(observation_ids)
                        labels.extend([-100] * len(observation_ids))
                        probabilities.extend([0.] * len(observation_ids))
                    if event.get("stop") == "tool_limit" or turn == self.max_turns - 1:
                        event.setdefault("stop", "turn_limit")
                        break
                    header = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
                    tokens.extend(header)
                    labels.extend([-100] * len(header))
                    probabilities.extend([0.] * len(header))
                # Observations may exceed the context budget; remove only the
                # unsupervised suffix so action labels are never re-positioned.
                while labels and labels[-1] == -100:
                    tokens.pop()
                    labels.pop()
                    probabilities.pop()
                if not tokens or all(value == -100 for value in labels):
                    raise ValueError("trajectory has no trainable actions")
                rows.append(tokens)
                label_rows.append(labels)
                probability_rows.append(probabilities)
                prompts.append(prompt)
                completions.append(completion)
                trajectories.append(events)
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, device=device, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        old_log_probs = torch.zeros((len(rows), width - 1), device=device)
        for index, (row, label, probability) in enumerate(zip(rows, label_rows, probability_rows)):
            ids[index, :len(row)] = torch.tensor(row, device=device)
            labels[index, :len(row)] = torch.tensor(label, device=device)
            mask[index, :len(row)] = 1
            old_log_probs[index, :len(row) - 1] = torch.tensor(probability[1:], device=device)
        tensors = {"input_ids": ids, "labels": labels, "attention_mask": mask}
        tensors["old_log_probs"] = old_log_probs
        return RolloutBatch(tensors, prompts, completions,
                            torch.arange(len(samples), device=device).repeat_interleave(num_generations),
                            self.policy_version, trajectories).validate()

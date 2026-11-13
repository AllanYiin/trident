"""Optional local chat-completions app. No server starts on import."""
import json
import threading
import time
import uuid

import torch

from trident.data.pipeline.language import ChatTemplate
from .agent import parse_response


def create_chat_app(item, model_name="trident"):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    if item.tokenizer is None:
        raise ValueError("serving requires a tokenizer")
    app = FastAPI(title="Trident local language model")
    lock = threading.Lock()

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "trident"}]}

    @app.post("/v1/chat/completions")
    def complete(request: dict):
        if request.get("model", model_name) != model_name:
            raise HTTPException(status_code=404, detail="unknown model")
        if request.get("n", 1) != 1:
            raise HTTPException(status_code=400, detail="this local server supports n=1")
        try:
            thinking = bool(request.get("open_thinking", False))
            text, _ = ChatTemplate().render(request["messages"], tools=request.get("tools"),
                                            add_generation_prompt=True, thinking=thinking)
            ids = torch.tensor([item.tokenizer.encode(text, add_special_tokens=False)], device=item.device, dtype=torch.long)
            max_tokens = int(request.get("max_tokens", 128))
            if max_tokens < 1:
                raise ValueError("max_tokens must be positive")
            with lock:
                output = item.generate(ids, max_new_tokens=max_tokens,
                                       temperature=float(request.get("temperature", 0.8)),
                                       top_p=float(request.get("top_p", 1.0)),
                                       eos_token_id=item.tokenizer.eos_token_id,
                                       pad_token_id=item.tokenizer.pad_token_id)
            completion = output[0, ids.shape[1]:].tolist()
            stopped = bool(completion and completion[-1] == item.tokenizer.eos_token_id)
            decoded = item.tokenizer.decode(completion[:-1] if stopped else completion, skip_special_tokens=False)
            parsed = parse_response(("<think>" if thinking else "") + decoded)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        message = {"role": "assistant", "content": parsed["content"]}
        if parsed["reasoning_content"] is not None:
            message["reasoning_content"] = parsed["reasoning_content"]
        if parsed["tool_calls"]:
            message["tool_calls"] = parsed["tool_calls"]
        finish = "tool_calls" if parsed["tool_calls"] else ("stop" if stopped else "length")
        common = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": model_name}
        usage = {"prompt_tokens": ids.shape[1], "completion_tokens": len(completion),
                 "total_tokens": ids.shape[1] + len(completion)}
        if not request.get("stream", False):
            return dict(common, object="chat.completion", usage=usage,
                        choices=[{"index": 0, "message": message, "finish_reason": finish}])

        def stream():
            # Buffered SSE preserves structured thinking/tool parsing. This
            # endpoint intentionally does not claim token-by-token latency.
            deltas = [{"role": "assistant"}]
            for key in ("reasoning_content", "content", "tool_calls"):
                if key in message:
                    value = message[key]
                    if key == "tool_calls":
                        value = [dict(call, index=index) for index, call in enumerate(value)]
                    deltas.append({key: value})
            for delta in deltas:
                event = dict(common, object="chat.completion.chunk",
                             choices=[{"index": 0, "delta": delta, "finish_reason": None}])
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps(dict(common, object="chat.completion.chunk", usage=usage,
                                             choices=[{"index": 0, "delta": {}, "finish_reason": finish}])) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream")

    return app

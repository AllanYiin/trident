"""Map LLM configuration names to Trident functions and layer classes."""

_LLM_ACTIVATIONS = {
    'swiglu': ('swiglu', 'SwiGLU'),
    'geglu': ('geglu', 'GeGLU'),
    'gpt_oss_swiglu': ('gpt_oss_swiglu', 'GptOssSwiGLU'),
    'gelu_pytorch_tanh': ('gpt_gelu', 'GptGelu'),
    'gelu_new': ('gpt_gelu', 'GptGelu'),
    'relu2': ('squared_relu', 'SquaredRelu'),
}


def llm_activation_name(name, only_layer):
    entry = _LLM_ACTIVATIONS.get(name.lower())
    if entry is None:
        return name
    function, layer = entry
    return layer if only_layer or name != name.lower() else function

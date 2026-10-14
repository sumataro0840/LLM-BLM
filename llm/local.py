"""
The same question answered by a model that runs on this machine (Hugging Face transformers, GPU).
torch and transformers are imported here only, so the OpenAI backend works without them.
"""
import copy
import ctypes
import re
import sys
from pathlib import Path

# The answer is one short number; generation stops here when the model keeps writing
# (Phi-3 writes one digit per token and often goes on for 15 digits: 12 tokens keep 9 decimals)
MAX_NEW_TOKENS = 12
# Answers generated at the same time. Each one holds its own copy of the prompt in GPU memory
# (about 0.4 MB per token for Phi-3-mini), and all n at once do not fit in 8 GB
BATCH_SIZE = 10
# a number at the start of the answer (the model is told to output the value only)
NUMBER = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def disable_power_throttling():
    """Stop Windows from moving this process to the efficiency cores (generation gets about 5 times slower)."""
    if sys.platform != "win32":
        return
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    # ProcessPowerThrottling = 4; EXECUTION_SPEED is put under control and turned off
    state = PowerThrottlingState(1, 1, 0)
    kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))


def make_client(model_id):
    """Load the model with 4-bit quantization. Returns (tokenizer, model, revision of the weights)."""
    import torch
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    disable_power_throttling()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16))
    model.eval()
    # the downloaded files are stored under .../snapshots/<revision>/
    cached = try_to_load_from_cache(model_id, "config.json")
    revision = Path(cached).parent.name[:8] if isinstance(cached, str) else "unknown"
    return tokenizer, model, revision


def parse_answer(text):
    """The number the answer starts with; None when it starts with something else, is a percent,
    or has an exponent that was cut off."""
    text = text.strip()
    match = NUMBER.match(text)
    if match is None or text[match.end():].lstrip()[:1] in ('%', 'e', 'E'):
        return None
    return float(match.group())


def sample_expected_returns(client, model, system_prompt, user_prompt, n, temperature):
    """Ask the question once and take n independent answers.

    Returns (answers, model_version): answers has n values, None where the answer could not be parsed.
    """
    import torch

    tokenizer, lm, revision = client
    inputs = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True).to(lm.device)
    stop_tokens = torch.tensor(lm.generation_config.eos_token_id, device=lm.device).reshape(-1)
    texts = []
    with torch.inference_mode():
        # the prompt is read once; every answer continues from a copy of its state
        prompt = lm(**inputs, use_cache=True)
        for start in range(0, n, BATCH_SIZE):
            size = min(BATCH_SIZE, n - start)
            cache = copy.deepcopy(prompt.past_key_values)
            cache.batch_repeat_interleave(size)
            logits = prompt.logits[:, -1].expand(size, -1)
            mask = inputs["attention_mask"].expand(size, -1)
            tokens = []
            for _ in range(MAX_NEW_TOKENS):
                # plain sampling from the model's distribution (no top-k / top-p cut), as the OpenAI API does by default
                token = torch.multinomial(torch.softmax(logits.float() / temperature, dim=-1), 1)
                tokens.append(token)
                mask = torch.cat([mask, mask.new_ones(size, 1)], dim=1)
                logits = lm(input_ids=token, attention_mask=mask, past_key_values=cache, use_cache=True).logits[:, -1]
            for row in torch.cat(tokens, dim=1):
                stops = torch.isin(row, stop_tokens).nonzero()
                texts.append(tokenizer.decode(row[:stops[0, 0]] if len(stops) else row, skip_special_tokens=True))
    # with the revision of the weights, so that a later change of the model can be noticed
    return [parse_answer(text) for text in texts], f"{model} @{revision} (4-bit nf4)"

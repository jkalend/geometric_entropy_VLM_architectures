"""
Memory-efficient VLM inference for HEDGE using Hugging Face transformers.
Supports Qwen2.5-VL (primary) and Med-Gemma.
Uses bfloat16 for inference.
"""

from pathlib import Path

# Unsloth must be imported before transformers for optimizations (per Unsloth docs).
# If import fails (e.g. PyTorch 2.10 flex_attention duplicate template), we fall back to standard path.
try:
    import unsloth  # noqa: F401
except Exception:
    unsloth = None

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

MODEL_CONFIGS = {
    "qwen2.5-vl-7b": ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen"),
    "qwen3-vl-8b": ("Qwen/Qwen3-VL-8B-Instruct", "qwen"),
    "qwen3-vl-30b": ("Qwen/Qwen3-VL-30B-A3B-Instruct", "qwen3_moe"),
    "medgemma-4b-it": ("google/medgemma-4b-it", "gemma"),
}


def _load_image(path: str | Path) -> Image.Image:
    p = Path(path)
    if p.exists():
        return Image.open(p).convert("RGB")
    raise FileNotFoundError(path)


def _generate_qwen(model, processor, image, question: str, temperature: float, max_new_tokens: int = 256):
    """Qwen2.5-VL specific generation."""
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )
    response = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    logprobs = [0.0] * max(1, len(response.split()))
    return response.strip(), logprobs


def _generate_qwen3_moe_with_hidden_states(
    model, processor, image, question: str, temperature: float, max_new_tokens: int = 256
) -> tuple[str, list[float], list[torch.Tensor] | None]:
    """Qwen3-VL-30B (MoE) generation with hidden states."""
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}
    ]
    # Qwen3-VL uses apply_chat_template with tokenize=True as shown in HF examples
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    
    input_len = inputs.input_ids.shape[1]
    response = processor.decode(out.sequences[0][input_len:], skip_special_tokens=True)
    logprobs = [0.0] * max(1, len(response.split()))

    layer_states = None
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        try:
            last_step = out.hidden_states[-1]
            if last_step is not None:
                layer_states = []
                for layer_h in last_step:
                    h = layer_h[0, -1, :].float().cpu()  # (hidden_dim,)
                    layer_states.append(h)
        except (IndexError, AttributeError, TypeError) as e:
            import warnings
            warnings.warn(f"Could not extract hidden states: {e}")
    return response.strip(), logprobs, layer_states


def _generate_qwen_with_hidden_states(
    model, processor, image, question: str, temperature: float, max_new_tokens: int = 256
) -> tuple[str, list[float], list[torch.Tensor] | None]:
    """Qwen2.5-VL generation with hidden states. Returns (response, logprobs, layer_hidden_states)."""
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    response = processor.decode(out.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    logprobs = [0.0] * max(1, len(response.split()))

    layer_states = None
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        try:
            # hidden_states: tuple per step; each step has tuple per layer
            # Take last step (final token) hidden states across all layers
            last_step = out.hidden_states[-1]
            if last_step is not None:
                # last_step: tuple of (batch, seq, hidden_dim) per layer; seq=1 for last token
                layer_states = []
                for layer_h in last_step:
                    h = layer_h[0, -1, :].float().cpu()  # (hidden_dim,)
                    layer_states.append(h)
        except (IndexError, AttributeError, TypeError) as e:
            import warnings
            warnings.warn(f"Could not extract hidden states: {e}")
    return response.strip(), logprobs, layer_states


def _generate_gemma_with_hidden_states(
    model, processor, image, question: str, temperature: float, max_new_tokens: int = 256
) -> tuple[str, list[float], list[torch.Tensor] | None]:
    """MedGemma generation with hidden states. Returns (response, logprobs, layer_hidden_states)."""
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    try:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    except Exception:
        prompt = f"<image>\n{question}"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    
    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(out.sequences[0][input_len:], skip_special_tokens=True)
    logprobs = [0.0] * max(1, len(response.split()))

    layer_states = None
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        try:
            # hidden_states: tuple per step; each step has tuple per layer
            last_step = out.hidden_states[-1]
            if last_step is not None:
                layer_states = []
                for layer_h in last_step:
                    h = layer_h[0, -1, :].float().cpu()  # (hidden_dim,)
                    layer_states.append(h)
        except (IndexError, AttributeError, TypeError) as e:
            import warnings
            warnings.warn(f"Could not extract hidden states: {e}")
    return response.strip(), logprobs, layer_states


def _generate_gemma(model, processor, image, question: str, temperature: float, max_new_tokens: int = 256):
    """MedGemma specific generation."""
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    try:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    except Exception:
        # Fallback if chat template fails
        prompt = f"<image>\n{question}"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )
    # Decode only the generated part
    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(out[0][input_len:], skip_special_tokens=True)
    logprobs = [0.0] * max(1, len(response.split()))
    return response.strip(), logprobs


def generate_single(model, processor, image_path: str, question: str, temperature: float = 0.7, max_new_tokens: int = 256, model_type: str = "qwen"):
    """Generate one answer. Supports Qwen2.5-VL, Qwen3-VL, and MedGemma."""
    image = _load_image(image_path)
    if model_type == "gemma" or (hasattr(model, "generate") and "Gemma" in type(model).__name__):
        return _generate_gemma(model, processor, image, question, temperature, max_new_tokens)
    if model_type == "qwen3_moe":
        # Using the specific Qwen3-VL-30B generation logic (without hidden states)
        res, lp, _ = _generate_qwen3_moe_with_hidden_states(model, processor, image, question, temperature, max_new_tokens)
        return res, lp
    if hasattr(model, "generate") and "Qwen" in type(model).__name__:
        return _generate_qwen(model, processor, image, question, temperature, max_new_tokens)
    
    # Fallback: try generic vision2seq
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    try:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    except Exception:
        prompt = question
        if "<image>" not in prompt and "image" in getattr(processor, "model_input_names", []):
            # Generic heuristic: prepend <image> if not present
            prompt = "<image>\n" + prompt
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=temperature > 0)
    
    # Try to decode only the generated part if input_ids is in inputs
    if "input_ids" in inputs:
        response = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    else:
        response = processor.decode(out[0], skip_special_tokens=True)
    return response.strip(), [0.0] * max(1, len(response.split()))


def generate_answers_transformers(
    vqa_data: list[dict],
    model_name: str = "qwen2.5-vl-7b",
    n_answers_high: int = 10,
    min_temp: float = 0.1,
    max_temp: float = 1.0,
    device: str = "cuda",
) -> list[dict]:
    """Generate answers for clean + distorted images. Returns HEDGE-style result dicts."""
    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        raise ValueError(f"Unknown model: {model_name}. Valid models: {list(MODEL_CONFIGS.keys())}")
    
    model_id = cfg[0] if isinstance(cfg, tuple) else cfg
    model_type = cfg[1] if isinstance(cfg, tuple) and len(cfg) > 1 else "qwen"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    if model_type == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    elif model_type == "qwen3_moe":
        # Qwen3-VL-30B-A3B is an MoE model; use 4-bit quantization to fit in VRAM
        from transformers import Qwen3VLMoeForConditionalGeneration, BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    model.eval()

    results = []
    for sample in tqdm(vqa_data, desc="Generating answers"):
        idx = sample["idx"]
        question = sample["question"]
        true_answer = sample["answer"]
        desc = sample.get("description")
        orig_path = sample["image_path"]
        distorted_paths = sample["distorted_image_paths"]

        base_text, base_log = generate_single(model, processor, orig_path, question, temperature=min_temp, model_type=model_type)
        original_low_temp = {"ans": base_text, "logprob": base_log}

        original_high_temp = []
        for _ in range(n_answers_high):
            text, log = generate_single(model, processor, orig_path, question, temperature=max_temp, model_type=model_type)
            original_high_temp.append({"ans": text, "logprob": log})

        distorted_high_temp = []
        for dpath in distorted_paths:
            text, log = generate_single(model, processor, dpath, question, temperature=max_temp, model_type=model_type)
            distorted_high_temp.append({"ans": text, "logprob": log})

        results.append({
            "idx_img": idx,
            "question": question,
            "image": orig_path,
            "true_answer": true_answer,
            "description": desc,
            "original_high_temp": original_high_temp,
            "distorted_high_temp": distorted_high_temp,
            "original_low_temp": original_low_temp,
            "variant_name": "default",
        })
    return results


def generate_answers_with_layer_dynamics(
    vqa_data: list[dict],
    model_name: str = "qwen2.5-vl-7b",
    n_answers_high: int = 10,
    min_temp: float = 0.1,
    max_temp: float = 1.0,
    device: str = "cuda",
) -> list[dict]:
    """Generate answers with hidden states for layer-wise semantic dynamics. Returns HEDGE-style dicts with layer_hidden_states.
    Supports Qwen2.5-VL, Qwen3-VL-30B, and MedGemma."""
    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        raise ValueError(f"Unknown model: {model_name}. Valid models: {list(MODEL_CONFIGS.keys())}")
    
    model_id = cfg[0] if isinstance(cfg, tuple) else cfg
    model_type = cfg[1] if isinstance(cfg, tuple) and len(cfg) > 1 else "qwen"
    
    if model_type not in ("qwen", "gemma", "qwen3_moe"):
        raise NotImplementedError(
            f"Layer dynamics supports qwen, gemma, and qwen3_moe only, got {model_type}"
        )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    
    if model_type == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        def _gen(im, q, t):
            return _generate_qwen_with_hidden_states(
                model, processor, _load_image(im), q, temperature=t
            )
    elif model_type == "qwen3_moe":
        from transformers import Qwen3VLMoeForConditionalGeneration
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        def _gen(im, q, t):
            return _generate_qwen3_moe_with_hidden_states(
                model, processor, _load_image(im), q, temperature=t
            )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        def _gen(im, q, t):
            return _generate_gemma_with_hidden_states(
                model, processor, _load_image(im), q, temperature=t
            )
            
    model.eval()

    results = []
    for sample in tqdm(vqa_data, desc="Generating (layer dynamics)"):
        idx = sample["idx"]
        question = sample["question"]
        true_answer = sample["answer"]
        desc = sample.get("description")
        orig_path = sample["image_path"]
        distorted_paths = sample["distorted_image_paths"]

        base_text, base_log, _ = _gen(orig_path, question, min_temp)
        original_low_temp = {"ans": base_text, "logprob": base_log}

        original_high_temp = []
        for _ in range(n_answers_high):
            text, log, layer_states = _gen(orig_path, question, max_temp)
            original_high_temp.append({"ans": text, "logprob": log, "layer_hidden_states": layer_states})

        distorted_high_temp = []
        for dpath in distorted_paths:
            text, log, layer_states = _gen(dpath, question, max_temp)
            distorted_high_temp.append({"ans": text, "logprob": log, "layer_hidden_states": layer_states})

        results.append({
            "idx_img": idx,
            "question": question,
            "image": orig_path,
            "true_answer": true_answer,
            "description": desc,
            "original_high_temp": original_high_temp,
            "distorted_high_temp": distorted_high_temp,
            "original_low_temp": original_low_temp,
            "variant_name": "default",
        })
    return results

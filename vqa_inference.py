import torch
from PIL import Image
import torch
from PIL import Image
# Assuming you renamed "adapter changing.py" to "adapter_changing.py" (spaces in filenames are bad!)
from adapter_changing import apply_adapter
def run_vqa_inference(model_zoo, adapter_type, image_path, query,  memory_context=""):
    """
    Handles VQA and Captioning for SAR, Optical, FCC, etc.
    """
    print(f"ðŸ‘ï¸ VQA Task [{adapter_type}]: {query}")

    # 1. Select the correct adapter path from our config
    # We map the classifier's output ('sar', 'optical') to your config keys
    adapter_path = model_zoo.loader.adapters.get(adapter_type)
    
    # 2. Apply the adapter
    if adapter_path:
        apply_adapter(model_zoo.qwen_model, adapter_path)
    else:
        print(f"â„¹ï¸ No specific adapter for '{adapter_type}'. Using Base Model.")

    # 3. Prepare Image
    image = Image.open(image_path).convert("RGB")

    # 4. Prepare Prompt (Standard Unsloth/Qwen Chat Template)
    if memory_context.strip():
        text_content = f"Context from previous conversation:\n{memory_context}\n\nCurrent question: {query}"
    else:
        text_content = query
    
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": text_content}
        ]}
    ]
    
    # 5. Tokenize
    tokenizer = model_zoo.qwen_tokenizer
    input_text = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        tokenize=False
    )
    
    inputs = tokenizer(
        [image],
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # 6. Generate
    # Uses the parameters from your 'optical.py' file (temp 1.5 is high, reduced to 0.7 for stability)
    with torch.no_grad():
        outputs = model_zoo.qwen_model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,
            temperature=0.7, 
            do_sample=False # Deterministic usually better for VQA
        )
    
    # 7. Decode
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    return response.strip()

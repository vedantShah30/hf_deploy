import torch
import torchvision.transforms as T
from PIL import Image
from adapter_changing import apply_adapter

# Global Cache to prevent re-running Stage 1 for the same image
# Key: image_path, Value: (fcc_features_text, converted_rgb_image_path)
FCC_CACHE = {}

# ==============================================================================
# PROMPTS
# ==============================================================================

FCC_STAGE_1_PROMPT = """
You are analyzing a False Color Composite (FCC) satellite image. FCC colors do NOT represent natural colors—ignore all color-based meaning.
Describe only what is visually and unambiguously present, in a short output.
Focus strictly on: geometric shapes, boundaries, edges, patterns, textures.
Do NOT name objects or land-cover types unless the shape alone makes it certain.
Do NOT infer, assume, or guess. Output only brief feature-level observations.
"""

RGB_STAGE_2_TEMPLATE = """
You are analyzing an RGB image that was converted from an FCC image.

Inputs:
1. The RGB image (provided visually).
2. FCC feature summary: {fcc_features}

Task: Combine RGB geometric cues with the FCC feature summary to answer the user's request.

User Request: {user_query}

Rules:
• No hallucination, no assumptions.
• Use only evidence supported by geometry + FCC features.
• Ignore colors if they contradict the FCC summary.
"""

# ==============================================================================
# HELPER: STYLE TRANSFER (FCC -> RGB)
# ==============================================================================
def convert_fcc_to_rgb(model_zoo, image_path):
    """
    Uses the loaded StyleTransformer (fcc_net) to convert the image.
    Returns a PIL Image.
    """
    model = model_zoo.fcc_net
    if not model:
        print("⚠️ FCC Net not loaded. Using original image.")
        return Image.open(image_path).convert("RGB")

    print("🎨 Running Style Transfer (FCC -> RGB)...")
    
    # 1. Load and Preprocess
    original_img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((256, 256)), # Ensure this matches your training size
        T.ToTensor(),
    ])
    
    # Check device of the model to move input tensor accordingly
    device = next(model.parameters()).device
    img_tensor = transform(original_img).unsqueeze(0).to(device)

    # 2. Inference
    with torch.no_grad():
        generated_tensor = model(img_tensor)
    
    # 3. Postprocess (Tensor -> PIL)
    generated_tensor = torch.clamp(generated_tensor, 0, 1)
    output_img = T.ToPILImage()(generated_tensor.squeeze(0).cpu())
    
    return output_img

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def run_fcc_pipeline(model_zoo, image_path, user_query):
    """
    Executes the 2-Stage FCC analysis using the BASE Qwen Model (No Adapter).
    """
    print(f"🔬 FCC Pipeline Active for: {image_path}")

    # 1. FORCE BASE MODEL (Disable Adapters)
    print("   -> 🛡️ Disabling Adapters (Using Base Qwen-VL)")
    apply_adapter(model_zoo.qwen_model, None)
    
    # 2. Check Cache for Stage 1 (Features) & Conversion
    if image_path in FCC_CACHE:
        print("⚡ Using Cached FCC Features & Converted Image.")
        fcc_features, rgb_image = FCC_CACHE[image_path]
    else:
        # --- STAGE 1: FEATURE EXTRACTION ---
        print("🔍 Stage 1: Extracting Geometric Features (Base Model)...")
        
        # Load Original FCC
        fcc_image = Image.open(image_path).convert("RGB")
        
        # Run Qwen on FCC
        fcc_features = _generate_text(model_zoo, fcc_image, FCC_STAGE_1_PROMPT, max_tokens=256)
        
        # --- INTERMEDIATE: CONVERT TO RGB ---
        # We convert now so we can cache it
        rgb_image = convert_fcc_to_rgb(model_zoo, image_path)
        
        # Save to Cache
        FCC_CACHE[image_path] = (fcc_features, rgb_image)

    # --- STAGE 2: RGB SYNTHESIS ---
    print(f"🧠 Stage 2: Answering Query '{user_query}'...")
    
    final_prompt = RGB_STAGE_2_TEMPLATE.format(
        fcc_features=fcc_features, 
        user_query=user_query
    )
    
    # Run Qwen on the *Converted RGB* Image
    response = _generate_text(model_zoo, rgb_image, final_prompt, max_tokens=512)
    
    return response

def _generate_text(model_zoo, image_obj, prompt, max_tokens=512):
    """
    Internal helper to run Unsloth generation
    """
    model = model_zoo.qwen_model
    tokenizer = model_zoo.qwen_tokenizer
    
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(
        [image_obj],
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        use_cache=True,
        temperature=0.7,
        do_sample=False, 
    )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    return response.strip()
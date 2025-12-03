import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
from unsloth import FastVisionModel
from tensorflow.keras.models import load_model
from geo_ground import load_geoground_pipeline

# ==============================================================================
# 1. IMAGE CLASSIFIER ARCHITECTURE & LOADER
# ==============================================================================

@tf.keras.utils.register_keras_serializable(package="custom_layers")
class HistogramLayer(tf.keras.layers.Layer):
    def __init__(self, nbins=256, value_range=(0.0, 255.0), **kwargs):
        super()._init_(**kwargs)
        self.nbins = int(nbins)
        self.value_range = (float(value_range[0]), float(value_range[1]))

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        gray = tf.image.rgb_to_grayscale(x)

        def per_image(img):
            img = tf.reshape(img, [-1])
            hist = tf.histogram_fixed_width(img, self.value_range, nbins=self.nbins)
            hist = tf.cast(hist, tf.float32)
            s = tf.reduce_sum(hist)
            return tf.cond(s > 0, lambda: hist / s, lambda: hist)

        return tf.map_fn(per_image, gray, fn_output_signature=tf.float32)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"nbins": self.nbins, "value_range": self.value_range})
        return cfg
def load_classifier_model(path):
    print(f"Loading Classifier from {path}...")
    
    # --- CHANGE HERE: Wrap load_model in a CPU device context ---
    with tf.device('/cpu:0'):
        # Explicitly pass custom objects so TF knows how to reconstruct the layer
        model = tf.keras.models.load_model(
            path, 
            compile=False, 
            custom_objects={'HistogramLayer': HistogramLayer},
        )

        model.compile(
            optimizer='adam',  # Use whatever optimizer you need
            jit_compile=False  # Disable XLA compilation
        )
    # --- END CHANGE ---
    return model
# ==============================================================================
# 2. FCC STYLE TRANSFORMER ARCHITECTURE & LOADER
# ==============================================================================

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, emb, patch):
        super()._init_()
        self.proj = nn.Conv2d(in_ch, emb, kernel_size=patch, stride=patch)
    def forward(self, x):
        x = self.proj(x)
        B, E, H, W = x.shape
        tokens = x.flatten(2).transpose(1,2)
        return tokens, (H, W)

class PatchUnembed(nn.Module):
    def __init__(self, emb, out_ch, patch):
        super()._init_()
        self.deproj = nn.ConvTranspose2d(emb, out_ch, kernel_size=patch, stride=patch)
    def forward(self, tokens, hw):
        B, N, E = tokens.shape
        H, W = hw
        x = tokens.transpose(1,2).reshape(B, E, H, W)
        return self.deproj(x)

class StyleTransformer(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, emb=96, layers=3, heads=6, ff=256, patch=16, image_size=256):
        super()._init_()
        self.image_size = image_size
        self.patch = patch
        self.h_p = image_size // patch
        self.w_p = self.h_p

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, emb, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.patch_embed = PatchEmbed(emb, emb, patch)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb, nhead=heads,
                                                dim_feedforward=ff, dropout=0.1,
                                                activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.patch_unembed = PatchUnembed(emb, emb, patch)
        self.decoder = nn.Sequential(
            nn.Conv2d(emb, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        N = (self.h_p * self.w_p)
        self.pos_embed = nn.Parameter(torch.randn(1, N, emb) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        # Simple resize checks or interpolation could be added here if needed
        z = self.encoder(x)
        tokens, hw = self.patch_embed(z)
        tokens = tokens + self.pos_embed.to(tokens.device)
        out_tokens = self.transformer(tokens)
        z_hat = self.patch_unembed(out_tokens, hw)
        z_hat = F.interpolate(z_hat, size=(H, W), mode='bilinear', align_corners=False)
        out = self.decoder(z_hat)
        return out

def load_fcc_net(weights_path, device="cuda"):
    print(f"Loading FCC Model from {weights_path}...")
    model = StyleTransformer(image_size=256)
    # Load state dict safely
    state_dict = torch.load(weights_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


# ==============================================================================
# 3. GLOBAL LOADER CLASS
# ==============================================================================

class GlobalModelLoader:
    def __init__(self):
        self.classifier = None
        self.fcc_model = None
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.geoground_pipeline = None
        self.adapters = {}
        
    def load_all(self, config_paths):
        """
        config_paths: dict containing paths for models
        Example:
        {
            'classifier': 'path/to/keras',
            'fcc': 'path/to/pt',
            'qwen': 'unsloth/Qwen3...',
            'lora_sar': 'path/to/sar',
            ...
        }
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Load Classifier
        if 'classifier' in config_paths:
            self.classifier = load_classifier_model(config_paths['classifier'])

        # 2. Load FCC Model
        if 'fcc' in config_paths:
            self.fcc_model = load_fcc_net(config_paths['fcc'], device=device)

        # 3. Load Qwen (Base)
        # Note: We load the base model. Adapters are applied during inference.
        print("Loading Qwen-VL Base Model...")
        self.qwen_model, self.qwen_tokenizer = FastVisionModel.from_pretrained(
            config_paths.get('qwen', "unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit"),
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        
        # 4. Store Adapter Paths (Don't load them all into VRAM yet if using Unsloth's dynamic loading)
        self.adapters = {
            'sar': config_paths.get('lora_sar'),
            'optical': config_paths.get('lora_optical'),
            'fcc': config_paths.get('lora_fcc'),
            'infra': config_paths.get('lora_infra')
        }

        # 5. Load GeoGround
        print("Loading GeoGround Pipeline...")
        # 5. Load GeoGround
        print("Loading GeoGround Pipeline...")
        self.geoground_pipeline = load_geoground_pipeline()

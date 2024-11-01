# promptfix.py

import os
import time
import torch
import numpy as np
from PIL import Image, ImageOps
import einops
from einops import rearrange
from omegaconf import OmegaConf
import sys
import torch.nn as nn
from torch import autocast
import k_diffusion as K

# Add stable_diffusion to path
sys.path.append("./stable_diffusion")
from stable_diffusion.ldm.util import instantiate_from_config



def reset_cuda(init_flag=False):
    """
    Manage CUDA device initialization and reset.
    
    Args:
        init_flag (bool): True for initialization, False for reset only
        
    Returns:
        tuple: (success (bool), device (torch.device), message (str))
    """
    try:
        # Synchronize before checking availability
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if not torch.cuda.is_available():
            return False, None, "CUDA not available on this system"
            
        # Get the current device index (usually 0 for single GPU)
        device_idx = torch.cuda.current_device()
        device = torch.device(f"cuda:{device_idx}")
        
        # Common cleanup operations
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        if init_flag:
            # Force CUDA initialization
            torch.cuda.init()
            torch.cuda.set_device(device)
            message = f"CUDA initialized on device {device_idx}: {torch.cuda.get_device_name(device_idx)}"
            
        else:
            # Additional reset operations
            if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                torch.cuda.reset_accumulated_memory_stats()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            message = f"CUDA device {device_idx} reset successfully"
            
        # Print memory status
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)    # Convert to GB
        message += f"\nMemory status - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB"

        # Synchronize after operations
        torch.cuda.synchronize()

        if not torch.cuda.is_available():
            return False, None, "CUDA became unavailable after operations"

        return True, device, message
        
    except Exception as e:
        return False, None, f"CUDA operation failed: {str(e)}"

def load_model_from_config(config, ckpt, vae_ckpt=None):
    """Load model from checkpoint with memory optimizations"""
    model = instantiate_from_config(config.model)
    pl_sd = torch.load(ckpt, map_location="cpu")
    
    if 'state_dict' in pl_sd:
        pl_sd = pl_sd['state_dict']
    
    m, u = model.load_state_dict(pl_sd, strict=False)
    model = model.half()  # Convert to half precision
    return model

def initialize_model(config_path, ckpt_path, vae_ckpt=None):
    """Initialize model with retry logic and memory optimizations"""
    max_retries = 3
    current_try = 0
    
    while current_try < max_retries:
        try:
            reset_cuda()
            
            # Load config
            config = OmegaConf.load(config_path)
            
            # Initialize model
            model = load_model_from_config(config, ckpt_path, vae_ckpt)
            model = model.half().eval().cuda()
            
            # Enable memory optimizations
            if hasattr(model, 'enable_attention_slicing'):
                model.enable_attention_slicing()
            if hasattr(model, 'enable_sequential_cpu_offload'):
                model.enable_sequential_cpu_offload()
            if hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
                
            return model
            
        except RuntimeError as e:
            current_try += 1
            print(f"Initialization attempt {current_try} failed: {str(e)}")
            
            if current_try == max_retries:
                raise RuntimeError("Failed to initialize model after multiple attempts")
            
            reset_cuda()
            time.sleep(5)

class CFGDenoiser(nn.Module):
    """Classifier-Free Guidance Denoiser"""
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale, input_image=None):
        if input_image is not None:
            z = self.hf_guidance(z, sigma, cond, x_original=input_image)
            
        cfg_z = einops.repeat(z, "b ... -> (repeat b) ...", repeat=3)
        cfg_sigma = einops.repeat(sigma, "b ... -> (repeat b) ...", repeat=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([
                cond["c_crossattn"][0], 
                uncond["c_crossattn"][0], 
                cond["c_crossattn"][0]
            ])],
            "c_concat": [torch.cat([
                cond["c_concat"][0], 
                cond["c_concat"][0], 
                uncond["c_concat"][0]
            ])],
        }
        
        out_cond, out_img_cond, out_txt_cond = self.inner_model(
            cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        
        return (0.5 * (out_img_cond + out_txt_cond) + 
                text_cfg_scale * (out_cond - out_img_cond) + 
                image_cfg_scale * (out_cond - out_txt_cond))

def resize_image_to_resolution(input_image, resolution, reverse=True):
    """Resize image while maintaining aspect ratio"""
    width, height = input_image.size
    scale = resolution / min(width, height) if reverse else resolution / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64
    return ImageOps.fit(input_image, (new_width, new_height), method=Image.Resampling.LANCZOS)

def process_image(model, image_path, output_path, task_prompt, context_prompt,
                 resolution=384, steps=10, cfg_text=6.5, cfg_image=1.25,
                 seed=2024, disable_hf_guidance=True, enable_flaw_prompt=True):
    """Process a single image using the PromptFix model"""
    try:
        # Load and prepare image
        input_image = Image.open(image_path).convert("RGB")
        input_image = resize_image_to_resolution(input_image, resolution)
        
        with autocast("cuda"):
            with torch.cuda.amp.autocast():
                # Prepare conditioning
                cond = {}
                if enable_flaw_prompt:
                    cond["c_crossattn"] = [model.get_learned_conditioning(
                        [task_prompt, context_prompt])]
                else:
                    cond["c_crossattn"] = [model.get_learned_conditioning(
                        [task_prompt])]
                
                # Convert image to tensor
                input_tensor = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                input_tensor = rearrange(input_tensor, "h w c -> 1 c h w").to(
                    next(model.parameters()).device)
                input_tensor = input_tensor.half()
                
                # Encode image
                cond["c_concat"] = [model.encode_first_stage(input_tensor).mode()]
                
                # Prepare unconditioned
                null_token = model.get_learned_conditioning([""])
                uncond = {
                    "c_crossattn": [torch.cat([null_token, null_token], 0) 
                                  if enable_flaw_prompt else null_token],
                    "c_concat": [torch.zeros_like(cond["c_concat"][0])]
                }
                
                # Setup denoiser
                model_wrap = K.external.CompVisDenoiser(model)
                model_wrap_cfg = CFGDenoiser(model_wrap)
                
                # Process image
                with torch.no_grad():
                    sigmas = model_wrap.get_sigmas(steps)
                    _, skip_connect_hs = model.first_stage_model.encoder(input_tensor)
                    z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                    
                    extra_args = {
                        "cond": cond,
                        "uncond": uncond,
                        "text_cfg_scale": cfg_text,
                        "image_cfg_scale": cfg_image,
                    }
                    
                    if not disable_hf_guidance:
                        extra_args["input_image"] = input_tensor
                    
                    z = K.sampling.sample_euler_ancestral(
                        model_wrap_cfg, z, sigmas, extra_args=extra_args)
                    x = model.decode_first_stage(z, skip_connect_hs=skip_connect_hs)
                
                # Convert to image
                x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                x = 255.0 * rearrange(x, "1 c h w -> h w c")
                edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
                
                # Save result
                edited_image.save(output_path)
                
                # Clear cache
                torch.cuda.empty_cache()
                
    except Exception as e:
        raise Exception(f"Image processing failed: {str(e)}")
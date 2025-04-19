import os
import sys
import torch
import tempfile
from PIL import Image
import numpy as np

# Function to generate image from text prompt
def generate_image_from_text(text_prompt, output_image_path):
    """
    Generate image from text prompt using Stable Diffusion
    
    Args:
        text_prompt (str): Text prompt for image generation
        output_image_path (str): Path to save the generated image
        
    Returns:
        str: Path to generated image
    """
    try:
        # Import required libraries
        from diffusers import StableDiffusionPipeline
        import rembg
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-compatible GPU is required to run this plugin")
        
        # Setup model
        model_id = "CompVis/stable-diffusion-v1-4"
        device = "cuda"
        
        # Load the model
        print("Loading Stable Diffusion model...")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to(device)
        
        # Ensure prompt requests white background
        if "white background" not in text_prompt.lower():
            text_prompt += " with a completely pure white background"
        
        # Generate image
        print(f"Generating image from prompt: {text_prompt}")
        image = pipe(text_prompt).images[0]
        
        # Save temporary image
        temp_path = os.path.join(tempfile.gettempdir(), "temp_generated.png")
        image.save(temp_path)
        
        # Remove background using rembg
        print("Cleaning up background...")
        input_image = Image.open(temp_path)
        output_image = rembg.remove(input_image)  # Returns image with transparent background
        
        # Create white background
        white_bg = Image.new("RGBA", output_image.size, (255, 255, 255, 255))
        
        # Composite the foreground on white background
        final_image = Image.alpha_composite(white_bg, output_image)
        
        # Convert to RGB (remove alpha channel)
        final_image = final_image.convert("RGB")
        
        # Save final image
        final_image.save(output_image_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        print(f"Image successfully generated and saved to: {output_image_path}")
        return output_image_path
        
    except Exception as e:
        print(f"Error generating image from text: {str(e)}")
        raise

# This module is designed to be imported by __init__.py
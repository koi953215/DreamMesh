import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
import numpy as np
import cv2

def main():
    # ====== Config ======
    model_ckpt_path = "background_mesh_generation/checkpoints/worldbuilder-v1-st.safetensors"  # .safetensors file
    output_dir = Path("background_mesh_generation")
    prompt = "a peaceful garden with trees and a pond, 360VR HDRI style"
    num_steps = 50
    guidance_scale = 7.5
    output_name = "generated_hdri.png"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ====== Load the full model checkpoint ======
    pipe = StableDiffusionPipeline.from_single_file(
        model_ckpt_path,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    # ====== Run Inference ======
    print(f"Generating image for prompt: {prompt}")
    image = pipe(prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale).images[0]

    # ====== Save Output ======
    output_path = output_dir / output_name
    image.save(output_path)
    # image_np = np.array(image).astype(np.float32) / 255.0
    # cv2.imwrite(str(output_path), image_np)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()

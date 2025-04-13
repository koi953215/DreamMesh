import os
import sys
import torch
import numpy as np
from PIL import Image
import tempfile

# Get the current plugin directory
__plugin_dir__ = os.path.dirname(os.path.realpath(__file__))

# Ensure LGM-full directory is in path
__lgm_dir__ = os.path.join(__plugin_dir__, "LGM-full")
if __lgm_dir__ not in sys.path:
    sys.path.append(__lgm_dir__)

# Global variable to store model instance to avoid reloading
_pipeline = None

def load_model():
    """
    Load the LGM model and return pipeline instance
    """
    global _pipeline
    
    if _pipeline is not None:
        return _pipeline
    
    try:
        from diffusers import DiffusionPipeline
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-compatible GPU is required to run this plugin")
        
        # Load the model
        print("Loading AI model, this may take some time...")
        _pipeline = DiffusionPipeline.from_pretrained(
            __lgm_dir__,
            custom_pipeline=__lgm_dir__,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda")
        
        print("AI model loaded successfully")
        return _pipeline
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_mesh_from_image(image_path, output_path):
    """
    Generate 3D model from image and save to specified path
    
    Args:
        image_path (str): Path to input JPG image
        output_path (str): Path to save output 3D model
        
    Returns:
        str: Path to output file
    """
    try:
        # 1. Load model
        pipeline = load_model()
        
        # 2. Load and preprocess image
        print(f"Processing image: {image_path}")
        pil_image = Image.open(image_path)
        input_image = np.array(pil_image).astype(np.float32) / 255.0
        
        # 3. Generate mesh
        print("Generating 3D model...")
        # Empty prompt as we're generating based on image only
        input_prompt = ""
        result = pipeline(input_prompt, input_image)
        
        # 4. Save result
        print(f"Saving model to: {output_path}")
        pipeline.save_ply(result, output_path)
        
        return output_path
        
    except Exception as e:
        print(f"Error generating mesh: {str(e)}")
        raise

# Function to convert to OBJ format if needed
def convert_ply_to_obj(ply_path):
    """
    Convert PLY file to OBJ format (if needed)
    This function uses Blender's API for conversion
    
    Args:
        ply_path (str): Path to input PLY file
        
    Returns:
        str: Path to output OBJ file
    """
    import bpy
    
    # Clear all objects in current scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Import PLY file
    bpy.ops.import_mesh.ply(filepath=ply_path)
    
    # Get imported object
    obj = bpy.context.selected_objects[0]
    
    # Create OBJ output path
    obj_path = os.path.splitext(ply_path)[0] + ".obj"
    
    # Export as OBJ
    bpy.ops.export_scene.obj(
        filepath=obj_path,
        use_selection=True,
        use_materials=False
    )
    
    # Delete imported object
    bpy.ops.object.delete()
    
    return obj_path

# Code for testing
if __name__ == "__main__":
    # Test function
    test_image = "test.jpg"
    output_path = "./output.ply"
    
    try:
        result_path = generate_mesh_from_image(test_image, output_path)
        print(f"Successfully generated model: {result_path}")
    except Exception as e:
        print(f"Test failed: {str(e)}")
bl_info = {
    "name": "AI Image/Text to Mesh Generator",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > AI Mesh",
    "description": "Generate 3D meshes from JPG images or text prompts using AI",
    "warning": "Requires CUDA-compatible GPU",
    "doc_url": "",
    "category": "3D View",
}

import bpy
import os
import sys
import tempfile
from bpy.props import StringProperty, PointerProperty, EnumProperty, BoolProperty
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras.io_utils import ImportHelper

# Get the current plugin directory
__file_path__ = os.path.dirname(os.path.realpath(__file__))

# Add plugin directory to system path for module imports
if __file_path__ not in sys.path:
    sys.path.append(__file_path__)

# Try to import mesh generator module
try:
    from . import mesh_generator
    from . import text_to_image
except ImportError as e:
    print(f"Could not import modules: {e}")

# Define plugin properties
class AIMeshGenProperties(PropertyGroup):
    def update_input_type(self, context):
        # Clear other input when type changes
        if self.input_type == 'IMAGE':
            self.text_prompt = ""
        else:
            self.image_path = ""
        return None
    
    input_type: EnumProperty(
        name="Input Type",
        description="Select input type: image or text",
        items=[
            ('IMAGE', "Image", "Generate from JPG image"),
            ('TEXT', "Text", "Generate from text prompt")
        ],
        default='IMAGE',
        update=update_input_type
    )
    
    image_path: StringProperty(
        name="Image Path",
        description="Select a JPG image for processing",
        default="",
        maxlen=1024,
        subtype='FILE_PATH'
    )
    
    text_prompt: StringProperty(
        name="Text Prompt",
        description="Enter text prompt to generate image and model",
        default="",
        maxlen=1024
    )
    
    output_format: EnumProperty(
        name="Output Format",
        description="Select output 3D model format",
        items=[
            ('PLY', "PLY", "Polygon File Format"),
            ('OBJ', "OBJ", "Wavefront OBJ (requires conversion)")
        ],
        default='PLY'
    )

# Image selection operator
class AIMeshGen_OT_SelectImage(Operator, ImportHelper):
    bl_idname = "aimeshgen.select_image"
    bl_label = "Select JPG Image"
    bl_description = "Select a JPG image to generate 3D model"
    
    filter_glob: StringProperty(
        default="*.jpg;*.jpeg",
        options={'HIDDEN'},
    )
    
    def execute(self, context):
        # Check if file is JPG
        if not (self.filepath.lower().endswith('.jpg') or self.filepath.lower().endswith('.jpeg')):
            self.report({'ERROR'}, "Please select a JPG image")
            return {'CANCELLED'}
            
        context.scene.ai_mesh_gen.image_path = self.filepath
        return {'FINISHED'}

# Mesh generation operator
class AIMeshGen_OT_GenerateMesh(Operator):
    bl_idname = "aimeshgen.generate"
    bl_label = "Generate 3D Model"
    bl_description = "Generate 3D model from the selected image or text prompt"
    
    @classmethod
    def poll(cls, context):
        # Enable when image is selected or text prompt is entered
        props = context.scene.ai_mesh_gen
        if props.input_type == 'IMAGE':
            return props.image_path != ""
        else:
            return props.text_prompt != ""
    
    def execute(self, context):
        props = context.scene.ai_mesh_gen
        output_format = props.output_format.lower()
        
        # Create progress dialog
        wm = context.window_manager
        wm.progress_begin(0, 100)
        
        try:
            # Use temp directory for output
            temp_dir = tempfile.gettempdir()
            output_file = os.path.join(temp_dir, f"ai_generated_mesh.{output_format}")
            
            # Check input type
            if props.input_type == 'IMAGE':
                image_path = props.image_path
                
                if not os.path.exists(image_path):
                    self.report({'ERROR'}, "Image file doesn't exist")
                    wm.progress_end()
                    return {'CANCELLED'}
                
                wm.progress_update(10)
                self.report({'INFO'}, "Processing image...")
                
            else:  # TEXT input
                text_prompt = props.text_prompt
                
                if not text_prompt.strip():
                    self.report({'ERROR'}, "Please enter a text prompt")
                    wm.progress_end()
                    return {'CANCELLED'}
                
                wm.progress_update(10)
                self.report({'INFO'}, "Generating image from text...")
                
                # Generate image from text
                temp_image_path = os.path.join(temp_dir, "text_generated_image.jpg")
                
                # Call text_to_image module to generate image
                try:
                    from . import text_to_image
                    image_path = text_to_image.generate_image_from_text(text_prompt, temp_image_path)
                    
                    # Verify image was generated
                    if not os.path.exists(image_path):
                        self.report({'ERROR'}, "Failed to generate image from text")
                        wm.progress_end()
                        return {'CANCELLED'}
                        
                except Exception as e:
                    self.report({'ERROR'}, f"Error generating image from text: {str(e)}")
                    wm.progress_end()
                    return {'CANCELLED'}
                
                wm.progress_update(40)
                self.report({'INFO'}, "Image generated, now creating 3D model...")
            
            # At this point, image_path variable contains the path to the image
            # (either uploaded by user or generated from text)
            
            # Generate mesh from image
            wm.progress_update(60)
            self.report({'INFO'}, "Generating 3D model...")
            
            # Call mesh_generator to create mesh
            try:
                mesh_generator.generate_mesh_from_image(image_path, output_file)
            except Exception as e:
                self.report({'ERROR'}, f"Error generating mesh: {str(e)}")
                wm.progress_end()
                return {'CANCELLED'}
            
            wm.progress_update(80)
            self.report({'INFO'}, "Importing 3D model...")
            
            # Import generated mesh
            if output_format == 'ply':
                bpy.ops.import_mesh.ply(filepath=output_file)
            elif output_format == 'obj':
                bpy.ops.import_scene.obj(filepath=output_file)
            
            wm.progress_update(100)
            wm.progress_end()
            
            # Select and focus on imported object
            if len(context.selected_objects) > 0:
                obj = context.selected_objects[0]
                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                context.view_layer.objects.active = obj
                bpy.ops.view3d.view_selected(use_all_regions=False)
            
            self.report({'INFO'}, f"Successfully generated and imported 3D model")
            return {'FINISHED'}
            
        except Exception as e:
            wm.progress_end()
            self.report({'ERROR'}, f"Error generating 3D model: {str(e)}")
            print(f"Detailed error: {str(e)}")
            return {'CANCELLED'}

# UI panel class
class AIMeshGen_PT_Panel(Panel):
    bl_label = "AI Image/Text to Mesh"
    bl_idname = "AIMeshGen_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AI Mesh"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.ai_mesh_gen
        
        # Input type selection
        layout.label(text="Step 1: Select Input Type", icon='OUTLINER_OB_FONT')
        layout.prop(props, "input_type", expand=True)
        
        # Image selection section - only active when IMAGE type is selected
        box = layout.box()
        if props.input_type == 'IMAGE':
            box.label(text="Select JPG Image", icon='IMAGE_DATA')
            row = box.row()
            row.prop(props, "image_path", text="")
            row.operator("aimeshgen.select_image", icon='FILE_FOLDER', text="")
            
            # Display selected image info
            if props.image_path:
                filename = os.path.basename(props.image_path)
                box.label(text=f"Selected: {filename}")
            
            # Image requirements tips
            col = box.column()
            col.label(text="Image Requirements:", icon='INFO')
            col.label(text="• JPG format only")
            col.label(text="• Clean background required")
            col.label(text="• Object should be centered")
            col.label(text="• High resolution recommended")
            
        # Text input section - only active when TEXT type is selected
        else:
            box.label(text="Enter Text Prompt", icon='TEXT')
            box.prop(props, "text_prompt", text="")
            
            # Text prompt tips
            col = box.column()
            col.label(text="Prompt Tips:", icon='INFO')
            col.label(text="• Be specific")
            col.label(text="• Single object recommended")
        
        # Output options
        box = layout.box()
        box.label(text="Output Options:", icon='SETTINGS')
        box.prop(props, "output_format")
        
        # Generate button
        layout.separator()
        layout.label(text="Step 2: Generate 3D Model")
        
        row = layout.row()
        row.scale_y = 2.0
        
        # Button is enabled only when appropriate input is provided
        enabled = (props.input_type == 'IMAGE' and props.image_path != "") or \
                  (props.input_type == 'TEXT' and props.text_prompt != "")
        
        if enabled:
            row.operator("aimeshgen.generate", icon='MESH_DATA')
        else:
            # Use a different style for disabled button
            row.operator("aimeshgen.generate", icon='MESH_DATA', text="Please Provide Input First")
            row.enabled = False

# Register and unregister functions
classes = (
    AIMeshGenProperties,
    AIMeshGen_OT_SelectImage,
    AIMeshGen_OT_GenerateMesh,
    AIMeshGen_PT_Panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ai_mesh_gen = PointerProperty(type=AIMeshGenProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.ai_mesh_gen

if __name__ == "__main__":
    register()
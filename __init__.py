bl_info = {
    "name": "AI Image to Mesh Generator",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > AI Mesh",
    "description": "Generate 3D meshes from JPG images using AI",
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
except ImportError as e:
    print(f"Could not import mesh generator module: {e}")

# Define plugin properties
class AIMeshGenProperties(PropertyGroup):
    image_path: StringProperty(
        name="Image Path",
        description="Select a JPG image for processing",
        default="",
        maxlen=1024,
        subtype='FILE_PATH'
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
    
    show_preview: BoolProperty(
        name="Show Preview",
        description="Show image preview before generation",
        default=True
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
    bl_description = "Generate 3D model from the selected image"
    
    @classmethod
    def poll(cls, context):
        # Only enable when an image is selected
        return context.scene.ai_mesh_gen.image_path != ""
    
    def execute(self, context):
        props = context.scene.ai_mesh_gen
        image_path = props.image_path
        output_format = props.output_format.lower()
        
        if not os.path.exists(image_path):
            self.report({'ERROR'}, "Image file doesn't exist")
            return {'CANCELLED'}
        
        # Create progress dialog
        wm = context.window_manager
        wm.progress_begin(0, 100)
        wm.progress_update(10)
        self.report({'INFO'}, "Processing image...")
        
        try:
            # Call mesh generation function
            wm.progress_update(30)
            self.report({'INFO'}, "Generating 3D model...")
            
            # Use temp directory for output
            temp_dir = tempfile.gettempdir()
            output_file = os.path.join(temp_dir, f"ai_generated_mesh.{output_format}")
            
            # Generate mesh
            mesh_generator.generate_mesh_from_image(image_path, output_file)
            
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
    bl_label = "AI Image to Mesh"
    bl_idname = "AIMeshGen_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AI Mesh"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.ai_mesh_gen
        
        # Image selection section
        box = layout.box()
        box.label(text="Step 1: Select JPG Image", icon='IMAGE_DATA')
        row = box.row()
        row.prop(props, "image_path", text="")
        row.operator("aimeshgen.select_image", icon='FILE_FOLDER', text="")
        
        # Display selected image info
        if props.image_path:
            filename = os.path.basename(props.image_path)
            box.label(text=f"Selected: {filename}")
            
            # Image preview if needed
            if props.show_preview:
                # Preview implementation would go here
                pass
        
        # Image requirements tips
        info_box = layout.box()
        info_box.label(text="Image Requirements:", icon='INFO')
        col = info_box.column()
        col.label(text="• JPG format only")
        col.label(text="• Clean background required")
        col.label(text="• Object should be centered")
        col.label(text="• High resolution recommended")
        
        # Output options
        box = layout.box()
        box.label(text="Output Options:", icon='SETTINGS')
        box.prop(props, "output_format")
        box.prop(props, "show_preview")
        
        # Generate button
        layout.separator()
        layout.label(text="Step 2: Generate 3D Model")
        
        row = layout.row()
        row.scale_y = 2.0
        row.operator("aimeshgen.generate", icon='MESH_DATA')

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
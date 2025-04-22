bl_info = {
    "name": "AI Scene Generator",
    "author": "Ting-Hsuan Chen",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Scene Generator",
    "description": "Generate a 3D scene using AI from a text description",
    "category": "3D View",
}

import bpy
import os
import json
import time
import tempfile
import shutil
import requests
import threading
from bpy.props import StringProperty, IntProperty, PointerProperty, BoolProperty
from bpy.types import Operator, Panel, PropertyGroup, AddonPreferences

# Import selenium modules
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Import OpenAI API
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Path to the addon directory
addon_dir = os.path.dirname(os.path.realpath(__file__))

# Global variables
TEMP_DIR = tempfile.gettempdir()
JSON_FILE_PATH = os.path.join(TEMP_DIR, "scene_generated.json")
SCENE_FOLDER = os.path.join(TEMP_DIR, "Scene")
MODELS_FOLDER = os.path.join(TEMP_DIR, "3D_Models")
ACTIVE_DRIVER = None

# === Addon Preferences ===
class AISceneGeneratorPreferences(AddonPreferences):
    bl_idname = __name__

    openai_api_key: StringProperty(
        name="OpenAI API Key",
        description="Enter your OpenAI API key",
        default="",
    )
    
    huggingface_username: StringProperty(
        name="HuggingFace Username",
        description="Enter your HuggingFace username or email",
        default=""
    )
    
    huggingface_password: StringProperty(
        name="HuggingFace Password",
        description="Enter your HuggingFace password",
        default="",
        subtype='PASSWORD'
    )
    
    def draw(self, context):
        layout = self.layout
        
        # OpenAI API settings
        box = layout.box()
        box.label(text="OpenAI API Settings:")
        box.prop(self, "openai_api_key")
        
        # HuggingFace login settings
        box = layout.box()
        box.label(text="HuggingFace Login Settings:")
        box.prop(self, "huggingface_username")
        box.prop(self, "huggingface_password")
        
        # Installation checks
        box = layout.box()
        box.label(text="Dependencies Status:")
        
        if OPENAI_AVAILABLE:
            box.label(text="OpenAI: Installed ✓", icon='CHECKMARK')
        else:
            box.label(text="OpenAI: Not Installed ✗", icon='CANCEL')
            box.operator("scenegen.install_openai")
        
        if SELENIUM_AVAILABLE:
            box.label(text="Selenium: Installed ✓", icon='CHECKMARK')
        else:
            box.label(text="Selenium: Not Installed ✗", icon='CANCEL')
            box.operator("scenegen.install_selenium")

# === Install Operators ===
class SCENEGEN_OT_InstallOpenAI(Operator):
    bl_idname = "scenegen.install_openai"
    bl_label = "Install OpenAI"
    bl_description = "Install OpenAI Python package"
    
    def execute(self, context):
        import subprocess
        import sys
        
        python_exe = sys.executable
        subprocess.check_call([python_exe, "-m", "pip", "install", "openai"])
        self.report({'INFO'}, "OpenAI installed. Please restart Blender.")
        return {'FINISHED'}

class SCENEGEN_OT_InstallSelenium(Operator):
    bl_idname = "scenegen.install_selenium"
    bl_label = "Install Selenium"
    bl_description = "Install Selenium Python package"
    
    def execute(self, context):
        import subprocess
        import sys
        
        python_exe = sys.executable
        subprocess.check_call([python_exe, "-m", "pip", "install", "selenium"])
        self.report({'INFO'}, "Selenium installed. Please restart Blender.")
        return {'FINISHED'}

# === Properties ===
class SceneGenProperties(PropertyGroup):
    scene_prompt: StringProperty(
        name="Scene Description",
        description="Describe the scene (e.g., 'a forest with animals')",
        default="",
        maxlen=1000,
    )

    object_count: IntProperty(
        name="Object Count",
        description="Number of objects to generate",
        default=3,
        min=1,
        max=25
    )
    
    generating: BoolProperty(
        default=False
    )
    
    import_models: BoolProperty(
        name="Import Models",
        description="Import 3D models into the scene after generation",
        default=True
    )

# === Operators ===
class SCENEGEN_OT_GenerateJSON(Operator):
    bl_idname = "scenegen.generate_json"
    bl_label = "Generate Scene JSON"
    bl_description = "Generate a JSON file using OpenAI based on scene description"

    def execute(self, context):
        preferences = context.preferences.addons[__name__].preferences
        props = context.scene.scene_gen
        scene_desc = props.scene_prompt.strip()
        count = props.object_count

        if not scene_desc:
            self.report({'ERROR'}, "Scene description cannot be empty.")
            return {'CANCELLED'}
        
        if not preferences.openai_api_key:
            self.report({'ERROR'}, "OpenAI API key is not set. Please check the addon preferences.")
            return {'CANCELLED'}

        self.report({'INFO'}, "Generating JSON...")
        
        # Create openai client
        print("Using OpenAI API Key:", preferences.openai_api_key)
        client = openai.OpenAI(api_key=preferences.openai_api_key)
        
        # Load example JSON
        example_path = os.path.join(addon_dir, "example.json")
        with open(example_path, 'r') as f:
            example_json_format = f.read()
            
        # Create chat prompt
        chat_prompt = f"""
Please generate a JSON file with {count} objects based on the scene description:
"{scene_desc}"
Match the structure and field names exactly as shown in this example:
{example_json_format}
Each object should include a `prompt` field for text-to-image generation, describing the object in high-quality detail.
Make sure:
- Each prompt describes a **single, complete, and centered object**
- The object should be **clearly separated from the background** (object-level focus)
- Use words like "isolated," "on a plain white background," or "studio-lit" to ensure easy background removal
- Avoid describing scenes or background elements
- The object should have a **clear contour** and be **fully visible** (no parts cropped or occluded)
Only return JSON. No extra explanation.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": chat_prompt}],
                temperature=0.7,
            )

            json_result = response.choices[0].message.content
            
            # Save JSON to temp directory
            with open(JSON_FILE_PATH, 'w') as f:
                f.write(json_result)

            self.report({'INFO'}, f"Scene JSON saved to: {JSON_FILE_PATH}")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"OpenAI API Error: {e}")
            return {'CANCELLED'}

class SCENEGEN_OT_GenerateImages(Operator):
    bl_idname = "scenegen.generate_images"
    bl_label = "Generate Images"
    bl_description = "Generate images from the scene description using HuggingFace"

    def create_scene_folder(self):
        """Create a Scene folder for storing generated images"""
        if os.path.exists(SCENE_FOLDER):
            shutil.rmtree(SCENE_FOLDER)
        os.makedirs(SCENE_FOLDER)
        return SCENE_FOLDER

    def login_huggingface(self, username, password):
        """Login to HuggingFace with provided credentials"""
        options = webdriver.ChromeOptions()
        prefs = {
            'profile.default_content_setting_values': {
                'notifications': 2
            }
        }
        options.add_experimental_option('prefs', prefs)
        options.add_argument("disable-infobars")
        options.add_argument("--start-maximized")
        
        driver = webdriver.Chrome(options=options)
        
        url = "https://huggingface.co/login"
        driver.get(url)
        
        time.sleep(2)
        
        try:
            username_field = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.NAME, "username"))
            )
            password_field = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.NAME, "password"))
            )
            
            username_field.clear()
            username_field.send_keys(username)
            
            password_field.clear()
            password_field.send_keys(password)
            
            login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
            login_button.click()
            
            time.sleep(2)
            
            return driver
            
        except Exception as e:
            print(f"Error during login: {e}")
            driver.quit()
            return None

    def generate_images_from_json(self, driver, json_file_path, scene_folder):
        """Reads JSON file and generates images based on prompts"""
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        url = "https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell"
        driver.get(url)
        
        time.sleep(5)
        
        iframe = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "space-iframe"))
        )
        driver.switch_to.frame(iframe)
        
        for obj in data["objects"]:
            name = obj["name"]
            prompt = obj["prompt"]
            
            input_element = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'input[data-testid="textbox"]'))
            )
            
            input_element.clear()
            input_element.send_keys(prompt)
            
            run_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.lg.secondary.svelte-cmf5ev'))
            )
            run_button.click()
            
            time.sleep(5)
            
            img_element = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'img.svelte-1pijsyv'))
            )
            
            img_url = img_element.get_attribute('src')
            
            try:
                image_data = requests.get(img_url).content
                
                image_path = os.path.join(scene_folder, f"{name}.webp")
                with open(image_path, 'wb') as img_file:
                    img_file.write(image_data)
                
                time.sleep(3)
                
            except Exception as e:
                print(f"Error saving image {name}: {e}")
        
        return True

    def execute(self, context):
        global ACTIVE_DRIVER
        preferences = context.preferences.addons[__name__].preferences
        
        if not os.path.exists(JSON_FILE_PATH):
            self.report({'ERROR'}, "JSON file not found. Generate JSON first.")
            return {'CANCELLED'}
            
        if not preferences.huggingface_username or not preferences.huggingface_password:
            self.report({'ERROR'}, "HuggingFace credentials not set. Please check the addon preferences.")
            return {'CANCELLED'}
            
        self.report({'INFO'}, "Creating scene folder...")
        scene_folder = self.create_scene_folder()
        
        # Check if we already have an active driver
        if ACTIVE_DRIVER is None:
            self.report({'INFO'}, "Logging in to HuggingFace...")
            ACTIVE_DRIVER = self.login_huggingface(
                preferences.huggingface_username,
                preferences.huggingface_password
            )
        
        if ACTIVE_DRIVER:
            try:
                self.report({'INFO'}, "Generating images from JSON...")
                success = self.generate_images_from_json(ACTIVE_DRIVER, JSON_FILE_PATH, scene_folder)
                
                if success:
                    self.report({'INFO'}, "All images generated successfully!")
                    return {'FINISHED'}
                else:
                    self.report({'ERROR'}, "Failed to generate images.")
                    return {'CANCELLED'}
                    
            except Exception as e:
                self.report({'ERROR'}, f"Error during execution: {e}")
                # Clean up driver on error
                ACTIVE_DRIVER.quit()
                ACTIVE_DRIVER = None
                return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "Failed to login to HuggingFace. Please check your credentials.")
            return {'CANCELLED'}

class SCENEGEN_OT_Generate3DModels(Operator):
    bl_idname = "scenegen.generate_3d_models"
    bl_label = "Generate 3D Models"
    bl_description = "Convert generated images to 3D models using Stable Fast 3D"

    def process_images_to_3d(self, driver, input_folder_path, output_folder_path, json_file_path):
        """Upload each image to Stable Fast 3D, process it, and download the GLB file"""
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        name_mapping = {}
        for obj in data["objects"]:
            name_mapping[obj["name"]] = obj["name"]
        
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        elif os.path.exists(output_folder_path):
            shutil.rmtree(output_folder_path)
            os.makedirs(output_folder_path)
        
        url = "https://huggingface.co/spaces/stabilityai/stable-fast-3d"
        driver.get(url)
        
        time.sleep(5)
        
        webp_files = [os.path.join(input_folder_path, f) for f in os.listdir(input_folder_path) if f.endswith('.webp')]
        
        for webp_file in webp_files:
            file_name = os.path.basename(webp_file)
            base_name = os.path.splitext(file_name)[0]
            
            if base_name in name_mapping:
                object_name = name_mapping[base_name]
                output_glb_path = os.path.join(output_folder_path, f"{object_name}.glb")
            else:
                output_glb_path = os.path.join(output_folder_path, f"{base_name}.glb")
            
            try:
                iframe = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "space-iframe"))
                )
                driver.switch_to.frame(iframe)
                
                file_upload = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'input[type="file"]'))
                )
                
                file_upload.send_keys(os.path.abspath(webp_file))
                
                remove_bg_button = WebDriverWait(driver, 30).until(
                    EC.element_to_be_clickable((By.ID, "component-13"))
                )
                remove_bg_button.click()
                
                time.sleep(2)
                
                run_button = WebDriverWait(driver, 30).until(
                    EC.element_to_be_clickable((By.ID, "component-13"))
                )
                run_button.click()

                time.sleep(2)
                
                download_button = WebDriverWait(driver, 120).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'a[download][href*=".glb"] button'))
                )
                
                download_link = driver.find_element(By.CSS_SELECTOR, 'a[download][href*=".glb"]')
                download_url = download_link.get_attribute('href')
                
                response = requests.get(download_url)
                with open(output_glb_path, 'wb') as f:
                    f.write(response.content)
                
                driver.switch_to.default_content()
                driver.refresh()
                time.sleep(3)
                
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                try:
                    driver.switch_to.default_content()
                    driver.refresh()
                    time.sleep(5)
                except:
                    pass
        
        return True

    def login_huggingface(self, username, password):
        """Login to HuggingFace with provided credentials"""
        options = webdriver.ChromeOptions()
        prefs = {
            'profile.default_content_setting_values': {
                'notifications': 2
            }
        }
        options.add_experimental_option('prefs', prefs)
        options.add_argument("disable-infobars")
        options.add_argument("--start-maximized")
        
        driver = webdriver.Chrome(options=options)
        
        url = "https://huggingface.co/login"
        driver.get(url)
        
        time.sleep(2)
        
        try:
            username_field = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.NAME, "username"))
            )
            password_field = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.NAME, "password"))
            )
            
            username_field.clear()
            username_field.send_keys(username)
            
            password_field.clear()
            password_field.send_keys(password)
            
            login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
            login_button.click()
            
            time.sleep(5)
            
            return driver
            
        except Exception as e:
            print(f"Error during login: {e}")
            driver.quit()
            return None

    def execute(self, context):
        global ACTIVE_DRIVER
        preferences = context.preferences.addons[__name__].preferences
        
        if not os.path.exists(JSON_FILE_PATH):
            self.report({'ERROR'}, "JSON file not found. Generate JSON first.")
            return {'CANCELLED'}
            
        if not os.path.exists(SCENE_FOLDER) or not os.listdir(SCENE_FOLDER):
            self.report({'ERROR'}, "No images found. Generate images first.")
            return {'CANCELLED'}
            
        if not preferences.huggingface_username or not preferences.huggingface_password:
            self.report({'ERROR'}, "HuggingFace credentials not set. Please check the addon preferences.")
            return {'CANCELLED'}
            
        # Check if we already have an active driver
        if ACTIVE_DRIVER is None:
            self.report({'INFO'}, "Logging in to HuggingFace...")
            ACTIVE_DRIVER = self.login_huggingface(
                preferences.huggingface_username,
                preferences.huggingface_password
            )
        
        if ACTIVE_DRIVER:
            try:
                self.report({'INFO'}, "Processing images to 3D models...")
                success = self.process_images_to_3d(ACTIVE_DRIVER, SCENE_FOLDER, MODELS_FOLDER, JSON_FILE_PATH)
                
                if success:
                    self.report({'INFO'}, "All 3D models generated successfully!")
                    return {'FINISHED'}
                else:
                    self.report({'ERROR'}, "Failed to generate 3D models.")
                    return {'CANCELLED'}
                    
            except Exception as e:
                self.report({'ERROR'}, f"Error during execution: {e}")
                return {'CANCELLED'}
            finally:
                # Clean up driver after all operations
                if ACTIVE_DRIVER:
                    ACTIVE_DRIVER.quit()
                    ACTIVE_DRIVER = None
        else:
            self.report({'ERROR'}, "Failed to login to HuggingFace. Please check your credentials.")
            return {'CANCELLED'}

class SCENEGEN_OT_ImportModels(Operator):
    bl_idname = "scenegen.import_models"
    bl_label = "Import Models to Scene"
    bl_description = "Import generated 3D models into Blender scene"

    def execute(self, context):
        if not os.path.exists(JSON_FILE_PATH):
            self.report({'ERROR'}, "JSON file not found.")
            return {'CANCELLED'}
            
        if not os.path.exists(MODELS_FOLDER) or not os.listdir(MODELS_FOLDER):
            self.report({'ERROR'}, "No 3D models found. Generate 3D models first.")
            return {'CANCELLED'}
            
        # Load the JSON to get position data
        with open(JSON_FILE_PATH, 'r') as file:
            data = json.load(file)
            
        # Import each GLB model and place according to JSON positions
        for obj in data["objects"]:
            name = obj["name"]
            position = obj.get("position", {"x": 0, "y": 0, "z": 0})
            
            model_path = os.path.join(MODELS_FOLDER, f"{name}.glb")
            
            if os.path.exists(model_path):
                # Import the GLB file
                bpy.ops.import_scene.gltf(filepath=model_path)
                
                # Get the imported object (usually the last selected)
                if bpy.context.view_layer.objects.selected:
                    obj_import = bpy.context.view_layer.objects.selected[0]
                    
                    # Set the position from JSON
                    obj_import.location.x = position.get("x", 0)
                    obj_import.location.y = position.get("y", 0)
                    obj_import.location.z = position.get("z", 0)
                    
                    # Rename the object to match the JSON name
                    obj_import.name = name
                else:
                    self.report({'WARNING'}, f"Could not get imported object for {name}")
            else:
                self.report({'WARNING'}, f"Model file not found for {name}")
                
        self.report({'INFO'}, "All models imported successfully!")
        return {'FINISHED'}

class SCENEGEN_OT_RunFullProcess(Operator):
    bl_idname = "scenegen.run_full_process"
    bl_label = "Generate Complete Scene"
    bl_description = "Run the entire process from JSON to importing models"

    _timer = None
    _thread = None
    current_step = 0
    total_steps = 4
    process_complete = False
    error_message = ""
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            props = context.scene.scene_gen
            
            if self.process_complete:
                # Process is complete
                props.generating = False
                self.cancel(context)
                
                if self.error_message:
                    self.report({'ERROR'}, self.error_message)
                    return {'CANCELLED'}
                else:
                    self.report({'INFO'}, "Scene generation complete!")
                    return {'FINISHED'}
                    
            # Update progress in UI
            context.area.tag_redraw()
                
        return {'PASS_THROUGH'}
    
    def execute(self, context):
        props = context.scene.scene_gen
        
        if props.generating:
            self.report({'WARNING'}, "Generation already in progress")
            return {'CANCELLED'}
            
        props.generating = True
        self.current_step = 0
        self.process_complete = False
        self.error_message = ""
        
        # Start timer for modal
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        
        # Start processing thread
        self._thread = threading.Thread(target=self.run_process, args=(context,))
        self._thread.start()
        
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        
    def run_process(self, context):
        global ACTIVE_DRIVER
        try:
            # Step 1: Generate JSON
            self.current_step = 1
            bpy.ops.scenegen.generate_json('EXEC_DEFAULT')
            
            # Step 2: Generate Images
            self.current_step = 2
            bpy.ops.scenegen.generate_images('EXEC_DEFAULT')
            
            # Step 3: Generate 3D Models
            self.current_step = 3
            bpy.ops.scenegen.generate_3d_models('EXEC_DEFAULT')
            
            # Step 4: Import Models (optional)
            if context.scene.scene_gen.import_models:
                self.current_step = 4
                bpy.ops.scenegen.import_models('EXEC_DEFAULT')
                
            self.process_complete = True
            
        except Exception as e:
            self.error_message = str(e)
            self.process_complete = True
        finally:
            # Ensure driver is cleaned up when process completes
            if ACTIVE_DRIVER:
                ACTIVE_DRIVER.quit()
                ACTIVE_DRIVER = None

# === UI Panel ===
class SCENEGEN_PT_MainPanel(Panel):
    bl_label = "AI Scene Generator"
    bl_idname = "SCENEGEN_PT_MainPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Scene Generator"

    def draw(self, context):
        layout = self.layout
        props = context.scene.scene_gen
        preferences = context.preferences.addons[__name__].preferences

        # Check if dependencies are available
        if not OPENAI_AVAILABLE or not SELENIUM_AVAILABLE:
            box = layout.box()
            box.label(text="Missing Dependencies:", icon='ERROR')
            if not OPENAI_AVAILABLE:
                box.label(text="OpenAI not installed")
                box.operator("scenegen.install_openai")
            if not SELENIUM_AVAILABLE:
                box.label(text="Selenium not installed")
                box.operator("scenegen.install_selenium")
            return

        # Credentials Check
        credentials_ok = True
        if not preferences.openai_api_key:
            layout.label(text="OpenAI API key not set", icon='ERROR')
            credentials_ok = False
        if not preferences.huggingface_username or not preferences.huggingface_password:
            layout.label(text="HuggingFace login not set", icon='ERROR')
            credentials_ok = False
            
        if not credentials_ok:
            layout.operator("preferences.addon_show", text="Open Preferences").module = __name__
            layout.separator()

        # Scene Description
        layout.label(text="Scene Description:")
        layout.prop(props, "scene_prompt", text="")

        # Object Count
        row = layout.row()
        row.label(text="Number of objects:")
        row.prop(props, "object_count", text="")
        
        # Import option
        layout.prop(props, "import_models")

        # Generate buttons
        layout.separator()
        
        if props.generating:
            # Show progress
            progress_op = SCENEGEN_OT_RunFullProcess
            if progress_op.current_step > 0:
                box = layout.box()
                row = box.row()
                row.label(text=f"Progress: Step {progress_op.current_step} of {progress_op.total_steps}")
                
                # Progress bar
                progress = progress_op.current_step / progress_op.total_steps
                row = box.row()
                row.prop(context.scene, "frame_current", text="")
                row.enabled = False
                
                # Step labels
                col = box.column(align=True)
                steps = ["Generating JSON", "Generating Images", "Creating 3D Models", "Importing Models"]
                for i, step_name in enumerate(steps):
                    icon = 'CHECKMARK' if progress_op.current_step > i else 'BLANK1'
                    col.label(text=step_name, icon=icon)
        else:
            # Full process button
            row = layout.row()
            row.scale_y = 1.5
            row.operator("scenegen.run_full_process", icon='PLAY')
            
            # Individual step buttons
            box = layout.box()
            box.label(text="Individual Steps:")
            col = box.column(align=True)
            col.operator("scenegen.generate_json", icon='TEXT')
            col.operator("scenegen.generate_images", icon='IMAGE_DATA')
            col.operator("scenegen.generate_3d_models", icon='MESH_DATA')
            col.operator("scenegen.import_models", icon='IMPORT')

# === Register ===
classes = (
    AISceneGeneratorPreferences,
    SCENEGEN_OT_InstallOpenAI,
    SCENEGEN_OT_InstallSelenium,
    SceneGenProperties,
    SCENEGEN_OT_GenerateJSON,
    SCENEGEN_OT_GenerateImages,
    SCENEGEN_OT_Generate3DModels,
    SCENEGEN_OT_ImportModels,
    SCENEGEN_OT_RunFullProcess,
    SCENEGEN_PT_MainPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.scene_gen = PointerProperty(type=SceneGenProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.scene_gen

if __name__ == "__main__":
    register()

import torch
from diffusers import FluxPipeline
from gradio_client import Client,file
import shutil

from glob import glob

# login externally before running this script like:
# from huggingface_hub import login;login(token="your_api_token") 

import json
loaded = json.load(open("out/scene.json",'r'))

#prompt="a big blue bird flying side view, isolated on a plain white background, fully visible and centered"
for obj in loaded["objects"]:
    print(obj["name"],obj["prompt"])

    # Generate image  (flux api pipeline)
    client = Client("black-forest-labs/FLUX.1-schnell")
    imgpath = client.predict( prompt=obj["prompt"], seed=0, randomize_seed=True, width=256, height=256, num_inference_steps=4, api_name="/infer")[0]
    shutil.copy(imgpath,"out/%s_img.png"%obj["name"])
    print(imgpath)

    # Generate mesh  (instantmesh api pipeline)
    client = Client("TencentARC/InstantMesh")
    result = client.predict( input_image=file(imgpath), api_name="/check_input_image")
    result = client.predict( input_image=file(imgpath), do_remove_background=True, api_name="/preprocess")
    result = client.predict( input_image=file(imgpath), sample_steps=75, sample_seed=42, api_name="/generate_mvs")
    meshpath = client.predict( api_name="/make3d")#[1]
    print(meshpath)
    shutil.copy(meshpath[0],"out/%s.obj"%obj["name"])
    shutil.copy(meshpath[1],"out/%s.glb"%obj["name"])

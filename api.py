from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import base64
from waiting import wait, TimeoutExpired
from pydantic import BaseModel
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline


#----------FAST-API CONFIG----------#
app = FastAPI(title="ARTISM")
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

#----------STABLE-DIFFUSION INIT----------#
device = "cuda"
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to(device)

#----------UPSCALE INIT----------#
# upscale_device = "cuda"
# upscale_model = "stabilityai/stable-diffusion-x4-upscaler"
# upscale_pipe = StableDiffusionPipeline.from_pretrained(upscale_model)
# upscale_pipe.to(upscale_device)

#----------QUEUE CONFIG----------#
queue = []
timeout_seconds = 120
next_id = 0

def getNextId():
    global next_id
    id = next_id
    next_id += 1
    return id

#----------RESPONSE----------#
class ReturnObject(BaseModel):
    id: int
    prompt: str
    image: str

#----------API----------#
@app.get("/")
def generate(prompt: str):
    id = getNextId()
    queue.append(id)

    def isFirstInQueue(id):
        if id == queue[0]:
            return True
        return False

    try:
        wait(lambda: isFirstInQueue(id), timeout_seconds=timeout_seconds, waiting_for="response")
    except TimeoutExpired as e:
        queue.remove(id)
        return Response(e)
    
    with autocast(device):
        image = pipe(prompt, num_inference_steps=1, guidance_scale=6).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    image.save("image.png")

    imgstr = base64.b64encode(buffer.getvalue())

    # with autocast(upscale_device):
    #     upscaled_image = upscale_pipe(prompt, buffer)
    
    # upscaled_image.save(buffer, format="PNG")
    # imgstr = base64.b64encode(buffer.getvalue())

    # upscaled_image.save("image.png")

    queue.remove(id)

    return ReturnObject(id=id, prompt=prompt, image=imgstr)

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import base64
import time
from waiting import wait, TimeoutExpired
from pydantic import BaseModel
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


#----------FAST-API CONFIG----------#
app = FastAPI(title="STAI_AI_KAPTEIN")
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

#----------STABLE-DIFFUSION CONFIG----------#
device = "cuda"
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to(device)
steps = 1

#----------QUEUE CONFIG----------#
queue = []
timeout_seconds = 120
next_id = 0

def getNextId():
    global next_id
    id = next_id
    next_id += 1
    return id

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
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=6).images[0]

    image.save("image.png")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    queue.remove(id)

    return ReturnObject(id=id, prompt=prompt, image=imgstr)

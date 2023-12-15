from fastapi import FastAPI, Response
from fastapi_socketio import SocketManager
import datetime
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import base64
from waiting import wait, TimeoutExpired
from pydantic import BaseModel
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image
from RealESRGAN import RealESRGAN


#----------FAST-API CONFIG----------#
app = FastAPI(title="ARTISM")
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)
# socket = SocketManager(app=app, cors_allowed_origins=[], async_mode="asgi")

#----------STABLE-DIFFUSION INIT----------#
device = "cuda"
model_id = "Yntec/dreamshaper-8"
pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
pipe.to(device)

#----------UPSCALE INIT----------#
upscale_device = torch.device("cuda")
upscale_model = RealESRGAN(upscale_device, scale=4)
upscale_model.load_weights('weights/RealESRGAN_x4.pth', download=True)

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
    image_small: str
    image_large: str

# async def sendPrompt(id, prompt):
#     dateTime = datetime.datetime.now()
#     time = f"{dateTime.hour}:{dateTime.minute}"

#     await socket.emit("prompt", { "time": time, "prompt": prompt, "id": id })

#----------API----------#
@app.get("/")
async def generate(prompt: str):

    negative = ""

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
        image = pipe(prompt, negative_prompt=negative, num_inference_steps=25, guidance_scale=6).images[0]

    # await sendPrompt(id, prompt)

    buffer = BytesIO()
    
    upscaled_image = upscale_model.predict(image)
    upscaled_image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    buffer.seek(0)
    buffer.truncate(0)

    image.save(buffer, format="PNG")
    img_small = base64.b64encode(buffer.getvalue())

    queue.remove(id)

    return ReturnObject(id=id, prompt=prompt, image_small=img_small, image_large=imgstr)


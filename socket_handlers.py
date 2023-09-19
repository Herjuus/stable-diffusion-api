from api import socket_manager as sm
import datetime

@sm.on("prompt")
async def listenprompt(sid, *args, **kwargs):
    print(args)

async def emit(prompt): 
    currentTime = datetime.datetime.now()
    time = f"{currentTime.hour}:{currentTime.minute}"

    await sm.emit("prompt", { prompt, time })

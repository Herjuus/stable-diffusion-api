from api import app
import datetime

@app.sio.on("prompt")
async def listenprompt(sid, *args, **kwargs):
    print(args)

async def emit(prompt): 
    currentTime = datetime.datetime.now()
    time = f"{currentTime.hour}:{currentTime.minute}"

    await app.sio.emit("prompt", { prompt, time })

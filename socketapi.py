from fastapi import FastAPI, Response
from fastapi_socketio import SocketManager
import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ARTISM")
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)
socket = SocketManager(app=app, cors_allowed_origins="*", async_mode="asgi")

@app.get("/message")
async def sendMessage(name: str, message: str):
    await socket.emit("message", {"name": name, "message": message})

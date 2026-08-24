from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.lifespan import lifespan
from app.api.routes import router
from app.config import settings

app = FastAPI(
    title="Audio Deepfake Detection API",
    version="1.0.0",
    description="Production-grade audio deepfake & voice clone detection using Wav2Vec2 + AASIST ensemble.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

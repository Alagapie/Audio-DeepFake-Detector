import time

from fastapi import FastAPI

from app.config import logger
from app.dependencies import init_components
from app.utils.audio_utils import check_ffmpeg

_start_time = time.time()


async def lifespan(app: FastAPI):
    global _start_time
    _start_time = time.time()
    check_ffmpeg()
    logger.info("Starting up — loading models...")
    comps = await init_components()
    app.state.components = comps
    app.state.start_time = _start_time
    logger.info("All models loaded successfully")
    yield
    logger.info("Shutting down — unloading...")
    await comps.wav2vec2.unload()
    await comps.aasist.unload()

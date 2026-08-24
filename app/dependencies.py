from app.config import logger, settings
from app.inference.orchestrator import InferenceOrchestrator
from app.models.aasist_pipeline import AASISTPipeline
from app.models.meta_classifier import MetaClassifier
from app.models.wav2vec2_pipeline import Wav2Vec2Pipeline

_components = None


class AppComponents:
    def __init__(self):
        self.wav2vec2 = Wav2Vec2Pipeline()
        self.aasist = AASISTPipeline()
        self.meta_classifier = MetaClassifier(settings.meta_classifier_path)

    @property
    def orchestrator(self) -> InferenceOrchestrator:
        return InferenceOrchestrator(self.wav2vec2, self.aasist, self.meta_classifier)


async def init_components() -> AppComponents:
    global _components
    if _components is not None:
        return _components

    comps = AppComponents()

    await comps.wav2vec2.load()
    await comps.aasist.load()
    comps.meta_classifier.load()

    _components = comps
    return comps


async def get_orchestrator() -> InferenceOrchestrator:
    comps = await init_components()
    return comps.orchestrator

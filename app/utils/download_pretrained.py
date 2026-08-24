"""
Convenience wrapper — delegates to the full download + remap pipeline.
"""

import logging

from app.utils.download_aasist import run_pipeline

logger = logging.getLogger(__name__)


def download_aasist_weights() -> bool:
    try:
        run_pipeline(download=True, save=True)
        return True
    except Exception as e:
        logger.warning(f"AASIST download pipeline failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_aasist_weights()


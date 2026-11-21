from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .preprocess import decode_base64_image
from .recognizer import get_engine, get_parseq_engine, recognize_image_bytes

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

app = FastAPI(title="Snap2 Local Python OCR Engine", version="1.0.0")


class OcrRequest(BaseModel):
    image: str
    ocrProfile: str = "auto"
    captureMode: str = "auto"
    exportStages: bool = True
    exportRootDir: Optional[str] = None


@app.get("/health")
def health():
    engine = get_engine()
    parseq = get_parseq_engine()
    return {
        "status": "ok",
        "engine": "python-local-ocr",
        "offlineOnly": True,
        "trocrReady": engine.is_available(),
        "parseqReady": parseq.is_available(),
    }


@app.post("/ocr")
def run_ocr(req: OcrRequest):
    if not req.image or not req.image.strip():
        raise HTTPException(status_code=400, detail="Missing base64 image.")
    try:
        image_bytes = decode_base64_image(req.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc

    try:
        result = recognize_image_bytes(
            image_bytes=image_bytes,
            ocr_profile=req.ocrProfile,
            export_stages=req.exportStages,
            export_root_dir=req.exportRootDir,
            capture_mode=req.captureMode,
        )
        return {"success": True, **result}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Python OCR error: {exc}") from exc

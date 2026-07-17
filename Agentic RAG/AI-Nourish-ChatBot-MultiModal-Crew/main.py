"""
NourishBot FastAPI backend.

Serves the NourishBot frontend (static/) and exposes a single endpoint,
POST /api/analyze, that replaces the Gradio `analyze_food` callback.

ASSUMPTION: NourishBotRecipeCrew / NourishBotAnalysisCrew expose the same
interface used in the original Gradio app:
    NourishBotRecipeCrew(image_data=..., dietary_restrictions=...).crew().kickoff(inputs)
    NourishBotAnalysisCrew(image_data=...).crew().kickoff(inputs)
returning a CrewOutput with .to_dict(). If src/crew.py differs, share it and
I'll adjust the call site — nothing else in this file needs to change.

Run:
    pip install fastapi "uvicorn[standard]" python-multipart
    uvicorn main:app --reload
Then open http://127.0.0.1:8000
"""
import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.crew import NourishBotAnalysisCrew, NourishBotRecipeCrew

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("nourishbot")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
CHUNK_SIZE = 1024 * 1024

app = FastAPI(title="NourishBot API")

# Tighten allow_origins before deploying — "*" is convenient for local dev only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/analyze")
async def analyze_food(
    image: UploadFile = File(...),
    dietary_restrictions: str = Form(""),
    workflow_type: str = Form(...),
):
    if workflow_type not in ("recipe", "analysis"):
        raise HTTPException(400, "workflow_type must be 'recipe' or 'analysis'")

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, "Upload a JPEG, PNG, or WEBP image")

    # Unique filename per request so concurrent uploads never overwrite each
    # other (the original Gradio app reused a single "uploaded_image.jpg").
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    image_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"

    size = 0
    try:
        with image_path.open("wb") as f:
            while chunk := await image.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(400, "Image is larger than 10MB")
                f.write(chunk)
    except HTTPException:
        image_path.unlink(missing_ok=True)
        raise
    finally:
        await image.close()

    try:
        inputs = {
            "uploaded_image": str(image_path),
            "dietary_restrictions": dietary_restrictions,
            "workflow_type": workflow_type,
        }

        if workflow_type == "recipe":
            crew_instance = NourishBotRecipeCrew(
                image_data=str(image_path),
                dietary_restrictions=dietary_restrictions,
            )
        else:
            crew_instance = NourishBotAnalysisCrew(image_data=str(image_path))

        # crew.kickoff() is synchronous and CrewAI refuses to run it if it
        # detects an active asyncio event loop (which this route always has,
        # since it's `async def`). Running it in a worker thread sidesteps
        # that check entirely.
        crew_output = await asyncio.to_thread(crew_instance.crew().kickoff, inputs=inputs)
        final_output = crew_output.to_dict()
    except Exception as exc:  # keep the API contract stable for the frontend
        logger.exception("NourishBot workflow failed (workflow_type=%s)", workflow_type)
        raise HTTPException(500, f"NourishBot workflow failed: {exc}") from exc
    finally:
        image_path.unlink(missing_ok=True)

    return JSONResponse({"workflow_type": workflow_type, "result": final_output})


# Mounted after the API routes above — /api/* and / are matched first.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
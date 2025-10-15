import logging
import os
import tempfile
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, Header
from pydantic import BaseModel
from dotenv import load_dotenv

from utils.transcriber import transcribe_audio
from utils.summarizer import summarize_transcript
from db import init_db, SessionLocal, Job
from sqlalchemy.orm import Session
import uuid
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("meeting-summarizer")

load_dotenv()

app = FastAPI(title="Meeting Summarizer API", version="1.0.0")

raw_origins = os.getenv(
    "CORS_ORIGINS",
    # Include local dev + deployed frontend (update if frontend domain changes)
    "http://127.0.0.1:5173,http://localhost:5173,https://meeting-summarizer-w431.onrender.com",
)
origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SummaryResponse(BaseModel):
    transcript: str
    summary: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Meeting Summarizer API")
    init_db()


@app.get("/health")
async def health():
    provider = "gemini"
    has_google = bool(os.getenv("GOOGLE_API_KEY"))
    return {"status": "ok", "provider": provider, "google_api_key": bool(has_google)}


@app.get("/health/full")
async def health_full():
    """Deeper readiness check: DB connectivity + row count. Does NOT call external APIs heavily.
    Returns diagnostic info but no secrets.
    """
    provider = "gemini"
    has_google = bool(os.getenv("GOOGLE_API_KEY"))
    db_ok = False
    job_count = None
    try:
        with SessionLocal() as s:
            job_count = s.query(Job).count()
            db_ok = True
    except Exception as e:  # pragma: no cover
        logger.warning("Health full DB check failed: %s", e)
    return {
        "status": "ok" if (db_ok and has_google) else "degraded",
        "provider": provider,
        "google_api_key": has_google,
        "db_ok": db_ok,
        "job_count": job_count,
    }


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.getenv("BACKEND_API_KEY")
    if expected:
        if not x_api_key or x_api_key != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")


class JobOut(BaseModel):
    id: str
    filename: Optional[str]
    status: str
    transcript: Optional[str]
    summary: Optional[Dict[str, Any]]


@app.post("/summarize", response_model=SummaryResponse, dependencies=[Depends(require_api_key)])
async def summarize(file: UploadFile = File(...), db: Session = Depends(get_db)):
    allowed_ext = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
    content_type_ok = bool(file.content_type and file.content_type.startswith("audio/"))
    ext = os.path.splitext(file.filename or "")[1].lower()
    if not content_type_ok and ext not in allowed_ext:
        logger.warning("Invalid file type. content_type=%s, filename=%s", file.content_type, file.filename)
        raise HTTPException(status_code=400, detail="Please upload an audio file (mp3, wav, m4a, aac, ogg, flac).")

    try:
        suffix = os.path.splitext(file.filename or "audio")[1] or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            # Enforce file size limit
            max_mb = float(os.getenv("MAX_UPLOAD_MB", "25"))
            if len(contents) > max_mb * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"File too large. Max {int(max_mb)}MB")
            tmp.write(contents)
            tmp_path = tmp.name
        logger.info("Saved uploaded file to temporary path: %s", tmp_path)

        # Create job record
        job_id = uuid.uuid4().hex[:16]
        job = Job(id=job_id, filename=file.filename or None, status="processing")
        db.add(job)
        db.commit()

        try:
            transcript = transcribe_audio(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
                logger.info("Removed temporary file: %s", tmp_path)
            except OSError:
                logger.exception("Failed to remove temporary file: %s", tmp_path)

        if not transcript:
            job.status = "failed"
            db.add(job)
            db.commit()
            raise HTTPException(status_code=502, detail="Transcription failed. Check API keys, file format, and server logs for details.")

        summary = summarize_transcript(transcript)
        if not isinstance(summary, dict):
            job.status = "failed"
            db.add(job)
            db.commit()
            raise HTTPException(status_code=502, detail="Summarization failed. See server logs for details.")

        # Persist
        job.status = "completed"
        job.transcript = transcript
        import json as _json
        job.summary_json = _json.dumps(summary)
        db.add(job)
        db.commit()

        return SummaryResponse(transcript=transcript, summary=summary)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during summarize: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/jobs/{job_id}", response_model=JobOut, dependencies=[Depends(require_api_key)])
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    import json as _json
    summary = _json.loads(job.summary_json) if job.summary_json else None
    return JobOut(id=job.id, filename=job.filename, status=job.status, transcript=job.transcript, summary=summary)


@app.get("/jobs", response_model=List[JobOut], dependencies=[Depends(require_api_key)])
def list_jobs(limit: int = 20, db: Session = Depends(get_db)):
    q = db.query(Job).order_by(Job.created_at.desc()).limit(max(1, min(limit, 100)))
    rows = q.all()
    import json as _json
    out: List[JobOut] = []
    for job in rows:
        out.append(JobOut(
            id=job.id,
            filename=job.filename,
            status=job.status,
            transcript=job.transcript,
            summary=_json.loads(job.summary_json) if job.summary_json else None,
        ))
    return out


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

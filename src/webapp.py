"""로컬 전용 웹 UI: 브라우저에서 캡컷 자동 편집 파이프라인을 실행한다.
외부 네트워크에 노출하지 않고 이 PC(localhost)에서만 쓰는 용도라 127.0.0.1에만 바인딩한다.
"""
from __future__ import annotations

import os
import subprocess
import threading
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .pipeline import PipelineError, run_pipeline
from .shorts_pipeline import run_shorts_pipeline
from .utils import load_config, project_root, sanitize_draft_name

ROOT = project_root()
CACHE_DIR = os.path.join(ROOT, "cache")
INDEX_HTML = os.path.join(ROOT, "web", "index.html")

app = FastAPI(title="캡컷 자동 편집")

_jobs: dict = {}
_jobs_lock = threading.Lock()
_dialog_lock = threading.Lock()


def _pick_file_dialog() -> Optional[str]:
    """탐색기 같은 네이티브 파일 선택 창을 띄우고 고른 경로를 돌려준다."""
    import tkinter as tk
    from tkinter import filedialog

    with _dialog_lock:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            path = filedialog.askopenfilename(
                title="편집할 영상/오디오 선택",
                filetypes=[
                    ("영상/오디오 파일", "*.mp4 *.mov *.mkv *.avi *.m4v *.mp3 *.wav *.m4a *.aac"),
                    ("모든 파일", "*.*"),
                ],
            )
        finally:
            root.destroy()
        return path or None


class RunRequest(BaseModel):
    video_path: str
    name: Optional[str] = None
    cut_video: bool = True
    apply_cuts: bool = True


class ShortsRunRequest(BaseModel):
    video_path: str
    name: Optional[str] = None
    title: Optional[str] = None


class RevealRequest(BaseModel):
    path: str


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.post("/api/reveal-file")
def reveal_file(req: RevealRequest) -> dict:
    """탐색기에서 해당 파일이 있는 폴더를 열고 파일을 선택된 상태로 보여준다."""
    path = os.path.abspath(req.path)
    if not os.path.exists(path):
        raise HTTPException(400, f"파일을 찾을 수 없습니다: {path}")
    subprocess.Popen(["explorer", "/select,", path])
    return {"ok": True}


@app.get("/api/pick-file")
def pick_file() -> dict:
    path = _pick_file_dialog()
    return {"path": path}


@app.post("/api/jobs")
def create_job(req: RunRequest) -> dict:
    video_path = os.path.abspath(req.video_path.strip().strip('"'))
    if not os.path.exists(video_path):
        raise HTTPException(400, f"파일을 찾을 수 없습니다: {video_path}")

    draft_name = sanitize_draft_name(
        (req.name or "").strip() or (os.path.splitext(os.path.basename(video_path))[0] + "_자동편집")
    )

    job_id = uuid.uuid4().hex[:12]
    job = {"status": "running", "logs": [], "result": None, "error": None}
    with _jobs_lock:
        _jobs[job_id] = job

    def log(msg: str) -> None:
        with _jobs_lock:
            job["logs"].append(msg)

    def worker() -> None:
        try:
            cfg = load_config(os.path.join(ROOT, "config.yaml"))
            result = run_pipeline(
                video_path, draft_name, cfg, CACHE_DIR, log=log, cut_video=req.cut_video, apply_cuts=req.apply_cuts
            )
            with _jobs_lock:
                job["status"] = "done"
                job["result"] = result
        except PipelineError as e:
            with _jobs_lock:
                job["status"] = "error"
                job["error"] = str(e)
        except Exception as e:  # noqa: BLE001 - 웹 UI로 모든 실패 사유를 보여주기 위함
            with _jobs_lock:
                job["status"] = "error"
                job["error"] = f"예상치 못한 오류: {e}"

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/shorts-jobs")
def create_shorts_job(req: ShortsRunRequest) -> dict:
    video_path = os.path.abspath(req.video_path.strip().strip('"'))
    if not os.path.exists(video_path):
        raise HTTPException(400, f"파일을 찾을 수 없습니다: {video_path}")

    draft_name = sanitize_draft_name(
        (req.name or "").strip() or (os.path.splitext(os.path.basename(video_path))[0] + "_숏폼")
    )

    job_id = uuid.uuid4().hex[:12]
    job = {"status": "running", "logs": [], "result": None, "error": None}
    with _jobs_lock:
        _jobs[job_id] = job

    def log(msg: str) -> None:
        with _jobs_lock:
            job["logs"].append(msg)

    def worker() -> None:
        try:
            cfg = load_config(os.path.join(ROOT, "config.yaml"))
            result = run_shorts_pipeline(
                video_path, draft_name, cfg, CACHE_DIR, log=log, title=(req.title or "").strip() or None
            )
            with _jobs_lock:
                job["status"] = "done"
                job["result"] = result
        except PipelineError as e:
            with _jobs_lock:
                job["status"] = "error"
                job["error"] = str(e)
        except Exception as e:  # noqa: BLE001 - 웹 UI로 모든 실패 사유를 보여주기 위함
            with _jobs_lock:
                job["status"] = "error"
                job["error"] = f"예상치 못한 오류: {e}"

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return {
            "status": job["status"],
            "logs": list(job["logs"]),
            "result": job["result"],
            "error": job["error"],
        }


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()

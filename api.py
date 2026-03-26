import os
import httpx
from typing import List, Optional, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from skill_analyzer import SkillAnalyzer, AnalysisResult, RecommendationItem, JobListing


# Global analyzer init (performance directive: no per-request model init)
ANALYZER = SkillAnalyzer.from_taxonomy_file("taxonomy.json")

# Job listings are loaded once at startup (only used when job-search is enabled)
_ALL_JOBS: List[JobListing] = SkillAnalyzer.load_jobs("jobs.json")

# SECURE MOODLE CONFIGURATION
MOODLE_TOKEN = os.getenv("MOODLE_TOKEN")
# Replace with your actual Moodle URL if different
MOODLE_REST_URL = os.getenv("MOODLE_REST_URL", "https://your-moodle-site.com/webservice/rest/server.php")

app = FastAPI(title="inRebus Edu - Digital Learning Hub API", version="0.1.0")

# MVP open CORS policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Mode = Literal["cv", "questionnaire"]


class AnalyzeRequest(BaseModel):
    target_role: str = Field(..., min_length=2)
    mode: Mode
    cv_text: Optional[str] = None
    answers: Optional[List[str]] = None


class AnalyzeResponse(BaseModel):
    target_role: str
    extracted_skills: List[str]
    match_percentage: float
    skill_gaps: List[str]
    recommendations: List[RecommendationItem]


class JobSearchRequest(BaseModel):
    skills: Optional[List[str]] = None
    query: Optional[str] = None
    limit: int = Field(default=10, le=50)


class JobSearchResponse(BaseModel):
    jobs: List[JobListing]
    total: int = Field(description="Total number of jobs returned")


class MoodleRequest(BaseModel):
    wsfunction: str
    params: dict = {}


def _require_job_search():
    """Dependency to ensure job search is enabled via config."""
    if not getattr(config, "ENABLE_JOB_SEARCH", True):
        raise HTTPException(status_code=403, detail="Job search feature is disabled.")
    return True


@app.get("/api/roles", response_model=List[str])
def get_roles() -> List[str]:
    return ANALYZER.get_roles()


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_skills(req: AnalyzeRequest) -> AnalyzeResponse:
    if req.mode == "cv" and not req.cv_text:
        raise HTTPException(status_code=400, detail="cv_text is required for cv mode.")
    if req.mode == "questionnaire" and not req.answers:
        raise HTTPException(status_code=400, detail="answers are required for questionnaire mode.")
        
    user_input = req.cv_text if req.mode == "cv" else " ".join(req.answers)
    
    try:
        result = ANALYZER.analyze(target_role=req.target_role, user_input=user_input)
        return AnalyzeResponse(
            target_role=result.target_role,
            extracted_skills=result.extracted_skills,
            match_percentage=result.match_percentage,
            skill_gaps=result.skill_gaps,
            recommendations=result.recommendations,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Role '{req.target_role}' not found in taxonomy.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/jobs",
    response_model=JobSearchResponse,
    dependencies=[Depends(_require_job_search)],
)
def list_jobs() -> JobSearchResponse:
    """Return all available job listings (no personalisation)."""
    return JobSearchResponse(jobs=_ALL_JOBS, total=len(_ALL_JOBS))


@app.post(
    "/api/jobs/search",
    response_model=JobSearchResponse,
    dependencies=[Depends(_require_job_search)],
)
def search_jobs(payload: JobSearchRequest) -> JobSearchResponse:
    """Return job listings ranked by match with the provided skill set."""
    user_skills: List[str] = payload.skills or []
    matched = ANALYZER.search_jobs(
        jobs=_ALL_JOBS,
        user_skills=user_skills,
        query=payload.query,
        limit=payload.limit,
    )
    return JobSearchResponse(jobs=matched, total=len(matched))


@app.post("/api/moodle/proxy")
async def moodle_proxy(payload: MoodleRequest):
    """
    Secure proxy for Moodle Web Services. 
    Hides the admin token from the client-side JavaScript.
    """
    if not MOODLE_TOKEN:
        raise HTTPException(status_code=500, detail="Moodle token not configured on server.")

    # SECURITY LOCKDOWN: Only allow specific, safe Moodle functions to be called.
    ALLOWED_FUNCTIONS = [
        "core_course_get_courses",
        "core_course_get_contents",
        "core_course_get_categories"
    ]
    
    if payload.wsfunction not in ALLOWED_FUNCTIONS:
        raise HTTPException(status_code=403, detail=f"Unauthorized Moodle function: {payload.wsfunction}")

    # Build the exact parameters Moodle expects
    moodle_params = {
        "wstoken": MOODLE_TOKEN,
        "wsfunction": payload.wsfunction,
        "moodlewsrestformat": "json"
    }
    # Merge in the parameters sent from the frontend (e.g., courseid)
    moodle_params.update(payload.params)

    async with httpx.AsyncClient() as client:
        try:
            # Note: Moodle REST API typically expects POST data as form-encoded, not raw JSON.
            response = await client.post(MOODLE_REST_URL, data=moodle_params)
            response.raise_for_status()
            
            # Moodle sometimes returns 200 OK but includes an "exception" in the JSON payload
            data = response.json()
            if isinstance(data, dict) and "exception" in data:
                print(f"Moodle API Exception: {data}")
                raise HTTPException(status_code=400, detail=data.get("message", "Moodle API Error"))
                
            return data
            
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Moodle communication failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    # Make sure your host is 0.0.0.0 for Render deployments
    uvicorn.run("api:app", host="0.0.0.0", port=10000, reload=True)

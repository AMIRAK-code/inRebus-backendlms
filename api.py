import os
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


@app.get("/api/roles", response_model=List[str])
def get_roles() -> List[str]:
    return ANALYZER.get_roles()


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    if payload.mode == "cv":
        if not payload.cv_text or not payload.cv_text.strip():
            raise HTTPException(status_code=422, detail="mode='cv' requires a non-empty cv_text.")
        user_input = payload.cv_text
    else:
        if not payload.answers or len(payload.answers) == 0:
            raise HTTPException(status_code=422, detail="mode='questionnaire' requires a non-empty answers array.")
        user_input = " ".join([a for a in payload.answers if isinstance(a, str)])
        if not user_input.strip():
            raise HTTPException(status_code=422, detail="answers must contain at least one non-empty string.")

    try:
        result: AnalysisResult = ANALYZER.analyze(target_role=payload.target_role, user_input=user_input)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown target_role: '{payload.target_role}'")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Unexpected server error during analysis.")

    return AnalyzeResponse(
        target_role=result.target_role,
        extracted_skills=result.extracted_skills,
        match_percentage=result.match_percentage,
        skill_gaps=result.skill_gaps,
        recommendations=result.recommendations,
    )


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

class FeatureFlagsResponse(BaseModel):
    job_search: bool = Field(description="Whether the job-search feature is active for this instance")


@app.get("/api/feature-flags", response_model=FeatureFlagsResponse)
def get_feature_flags() -> FeatureFlagsResponse:
    """Return which optional features are enabled on this instance.

    Clients should call this endpoint on startup to conditionally render UI
    elements (e.g. hide the job-search tab when the feature is disabled).
    """
    return FeatureFlagsResponse(job_search=config.ENABLE_JOB_SEARCH)


# ---------------------------------------------------------------------------
# Job-search feature (feature-flagged)
# ---------------------------------------------------------------------------

def _require_job_search() -> None:
    """FastAPI dependency that enforces the job-search feature flag.

    Raises HTTP 403 when the ``ENABLE_JOB_SEARCH`` flag is ``False``, keeping
    job-search endpoints completely inaccessible for client instances that have
    not enabled the feature.
    """
    if not config.ENABLE_JOB_SEARCH:
        raise HTTPException(
            status_code=403,
            detail=(
                "The job-search feature is not enabled on this instance. "
                "Please contact your administrator if you need access to this feature."
            ),
        )


class JobSearchRequest(BaseModel):
    skills: Optional[List[str]] = Field(
        default=None,
        description="List of user skills to match against job requirements",
    )
    query: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Optional free-text search term to filter by job title or description",
    )
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results to return")


class JobSearchResponse(BaseModel):
    jobs: List[JobListing]
    total: int = Field(description="Total number of jobs returned")


@app.get(
    "/api/jobs",
    response_model=JobSearchResponse,
    dependencies=[Depends(_require_job_search)],
)
def list_jobs() -> JobSearchResponse:
    """Return all available job listings (no personalisation).

    Requires the ``ENABLE_JOB_SEARCH`` feature flag to be active.
    """
    return JobSearchResponse(jobs=_ALL_JOBS, total=len(_ALL_JOBS))


@app.post(
    "/api/jobs/search",
    response_model=JobSearchResponse,
    dependencies=[Depends(_require_job_search)],
)
def search_jobs(payload: JobSearchRequest) -> JobSearchResponse:
    """Return job listings ranked by match with the provided skill set.

    Requires the ``ENABLE_JOB_SEARCH`` feature flag to be active.

    - **skills**: provide the ``extracted_skills`` from ``POST /api/analyze``
      to get personalised job recommendations.
    - **query**: optional keyword to restrict results by job title/description.
    - **limit**: cap the number of returned results (default 10, max 50).
    """
    user_skills: List[str] = payload.skills or []
    matched = SkillAnalyzer.search_jobs(
        jobs=_ALL_JOBS,
        user_skills=user_skills,
        query=payload.query,
        limit=payload.limit,
    )
    return JobSearchResponse(jobs=matched, total=len(matched))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info",
    )
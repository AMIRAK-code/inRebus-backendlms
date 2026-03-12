import os
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from skill_analyzer import SkillAnalyzer, AnalysisResult, RecommendationItem


# Global analyzer init (performance directive: no per-request model init)
ANALYZER = SkillAnalyzer.from_taxonomy_file("taxonomy.json")

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info",
    )
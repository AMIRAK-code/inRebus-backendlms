import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_WORD_RE = re.compile(r"[A-Za-z0-9\-\+\.#]+")


def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    tokens = _WORD_RE.findall(s)
    return " ".join(tokens)


def _safe_float(x: float) -> float:
    if x is None or np.isnan(x):
        return 0.0
    return float(x)


class RecommendationItem(BaseModel):
    id: int
    title: str
    target_skill: str
    type: Literal["Course", "Article", "Video", "Project"] | str
    duration: str


@dataclass(frozen=True)
class AnalysisResult:
    target_role: str
    extracted_skills: List[str]
    match_percentage: float
    skill_gaps: List[str]
    recommendations: List[RecommendationItem]


class SkillAnalyzer:
    """
    MVP analyzer:
    - Builds TF-IDF on role documents (skills/knowledge text) at boot.
    - For requests: vectorizes user text once and compares to target role doc.
    - Extracted skills: token overlap between user input and role skills (fast heuristic).
    - Gaps: role skills not detected in user tokens.
    - Recommendations: simple stubs per gap (frontend-ready).
    """

    def __init__(self, taxonomy: Dict[str, Any]) -> None:
        self.taxonomy = taxonomy
        self.roles = self._extract_roles(taxonomy)  # role -> role_obj
        if not self.roles:
            raise ValueError("Taxonomy contains no roles/profiles.")

        self.role_names = sorted(self.roles.keys())

        role_docs: List[str] = []
        for role_name in self.role_names:
            role_docs.append(self._role_to_document(self.roles[role_name]))

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=40000,
        )
        self.role_matrix = self.vectorizer.fit_transform(role_docs)

        self._role_skills: Dict[str, List[str]] = {
            role: self._extract_skills_list(self.roles[role]) for role in self.role_names
        }

    @classmethod
    def from_taxonomy_file(cls, path: str) -> "SkillAnalyzer":
        with open(path, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)
        return cls(taxonomy)

    def get_roles(self) -> List[str]:
        return self.role_names

    def analyze(self, target_role: str, user_input: str) -> AnalysisResult:
        if target_role not in self.roles:
            raise KeyError(target_role)

        user_doc = _normalize_text(user_input)
        if not user_doc:
            raise ValueError("Empty user input after normalization.")

        user_vec = self.vectorizer.transform([user_doc])

        role_idx = self.role_names.index(target_role)
        role_vec = self.role_matrix[role_idx]

        sim = cosine_similarity(user_vec, role_vec)[0][0]
        match_percentage = round(_safe_float(sim) * 100.0, 2)

        role_skills = self._role_skills.get(target_role, [])
        extracted_skills, gaps = self._compute_skills_and_gaps(user_doc, role_skills)
        recommendations = self._recommend_for_gaps(gaps)

        return AnalysisResult(
            target_role=target_role,
            extracted_skills=extracted_skills,
            match_percentage=match_percentage,
            skill_gaps=gaps,
            recommendations=recommendations,
        )

    def _extract_roles(self, taxonomy: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        if isinstance(taxonomy.get("profiles"), list):
            out: Dict[str, Dict[str, Any]] = {}
            for p in taxonomy["profiles"]:
                name = (p.get("name") or "").strip()
                if name:
                    out[name] = p
            return out

        if isinstance(taxonomy.get("roles"), dict):
            out: Dict[str, Dict[str, Any]] = {}
            for name, obj in taxonomy["roles"].items():
                if isinstance(name, str) and name.strip() and isinstance(obj, dict):
                    out[name.strip()] = obj
            return out

        return {}

    def _extract_skills_list(self, role_obj: Dict[str, Any]) -> List[str]:
        skills = role_obj.get("skills") or []
        out: List[str] = []
        if isinstance(skills, list):
            for s in skills:
                if isinstance(s, str):
                    val = s.strip()
                    if val:
                        out.append(val)
                elif isinstance(s, dict):
                    val = (s.get("name") or "").strip()
                    if val:
                        out.append(val)
        return sorted(set(out))

    def _role_to_document(self, role_obj: Dict[str, Any]) -> str:
        parts: List[str] = []

        for s in self._extract_skills_list(role_obj):
            parts.append(s)

        knowledge = role_obj.get("knowledge") or []
        if isinstance(knowledge, list):
            for k in knowledge:
                if isinstance(k, str) and k.strip():
                    parts.append(k.strip())
                elif isinstance(k, dict):
                    val = (k.get("name") or "").strip()
                    if val:
                        parts.append(val)

        desc = role_obj.get("description")
        if isinstance(desc, str) and desc.strip():
            parts.append(desc.strip())

        return _normalize_text(" ".join(parts))

    def _compute_skills_and_gaps(self, user_doc: str, role_skills: List[str]) -> Tuple[List[str], List[str]]:
        user_tokens = set(user_doc.split())

        extracted: List[str] = []
        gaps: List[str] = []

        for skill in role_skills:
            skill_norm = _normalize_text(skill)
            skill_tokens = set(skill_norm.split())

            if skill_tokens and user_tokens.intersection(skill_tokens):
                extracted.append(skill)
            else:
                gaps.append(skill)

        return extracted[:25], gaps[:25]

    def _recommend_for_gaps(self, gaps: List[str]) -> List[RecommendationItem]:
        recs: List[RecommendationItem] = []
        for idx, skill in enumerate(gaps, start=1):
            recs.append(
                RecommendationItem(
                    id=idx,
                    title=f"Intro to {skill}",
                    target_skill=skill,
                    type="Course",
                    duration="4h",
                )
            )
        return recs
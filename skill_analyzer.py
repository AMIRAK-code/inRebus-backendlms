import json
import os
import csv
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
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
    upvotes: int = Field(default=0, ge=0, description="Community upvote count for this recommendation")


class JobListing(BaseModel):
    id: int
    title: str
    company: str
    location: str
    description: str
    required_skills: List[str]
    match_percentage: Optional[float] = Field(
        default=None,
        description="Percentage of required skills matched against the user's profile (0-100)",
    )


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
    """

    def __init__(self, taxonomy: Dict[str, Any]) -> None:
        self.taxonomy = taxonomy
        self.roles = self._extract_roles(taxonomy)
        
        # Safeguard to prevent server crash if taxonomy is empty
        if not self.roles:
            print("WARNING: Taxonomy is empty. Injecting fallback role to keep server alive.")
            self.roles = {"Fallback Role": {"name": "Fallback Role", "skills": ["system integration"]}}

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
        """Safely loads JSON, preventing crashes from Git LFS pointers."""
        taxonomy = {"profiles": []}
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                # Check if it's a Git LFS pointer instead of real JSON
                if content.startswith("version https://git-lfs"):
                    print(f"CRITICAL: {path} is a Git LFS pointer. Render did not download the real file.")
                elif content:
                    taxonomy = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"CRITICAL: JSON Decode Error in {path}: {e}")
        except FileNotFoundError:
            print(f"CRITICAL: {path} not found.")
            
        return cls(taxonomy)

    @classmethod
    def from_competenze_csv(cls, path: str) -> "SkillAnalyzer":
        """
        Directly parses a competencies CSV to bypass GitHub LFS JSON size limits.
        Supports both English and Italian taxonomy structures.
        """
        competencies = {}
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Dynamically check for either English or Italian column headers
                    comp_name = row.get("Competence Name", row.get("Competence", row.get("DENOMINAZIONE COMPETENZA", ""))).strip()
                    tipo = row.get("Type", row.get("TIPO", "")).strip().lower()
                    desc = row.get("Description", row.get("DESCRIZIONE ABILITA'/CONOSCENZA", "")).strip()
                    
                    if not comp_name or not desc:
                        continue
                        
                    if comp_name not in competencies:
                        competencies[comp_name] = {"name": comp_name, "skills": [], "knowledge": []}
                        
                    # Map to arrays based on English or Italian type strings
                    if "skill" in tipo or "abilità" in tipo:
                        competencies[comp_name]["skills"].append(desc)
                    elif "knowledge" in tipo or "conoscenz" in tipo:
                        competencies[comp_name]["knowledge"].append(desc)
                        
            taxonomy = {"profiles": list(competencies.values())}
            print(f"Successfully loaded {len(taxonomy['profiles'])} competencies directly from CSV.")
        except Exception as e:
            print(f"CSV Load Error: {e}")
            taxonomy = {"profiles": []}
            
        return cls(taxonomy)

    @classmethod
    def from_esco_json(cls, path: str) -> "SkillAnalyzer":
        """
        Directly parses flat ESCO JSON arrays into the internal taxonomy format.
        Maps occupationLabel -> role, skillLabel -> skills/knowledge.
        """
        competencies = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content.startswith("version https://git-lfs"):
                    print(f"CRITICAL: {path} is a Git LFS pointer.")
                    return cls({"profiles": []})
                
                raw_data = json.loads(content)
                
                for row in raw_data:
                    role_name = row.get("occupationLabel", "").strip()
                    skill_type = row.get("skillType", "").strip().lower()
                    skill_name = row.get("skillLabel", "").strip()
                    
                    if not role_name or not skill_name:
                        continue
                        
                    if role_name not in competencies:
                        competencies[role_name] = {"name": role_name, "skills": [], "knowledge": []}
                        
                    if "knowledge" in skill_type:
                        competencies[role_name]["knowledge"].append(skill_name)
                    else:
                        competencies[role_name]["skills"].append(skill_name)
                        
            taxonomy = {"profiles": list(competencies.values())}
            print(f"Successfully loaded {len(taxonomy['profiles'])} ESCO occupations.")
        except Exception as e:
            print(f"ESCO Load Error: {e}")
            taxonomy = {"profiles": []}
            
        return cls(taxonomy)

    # ------------------------------------------------------------------
    # Job-search helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_jobs(path: str) -> List[JobListing]:
        with open(path, "r", encoding="utf-8") as f:
            raw: List[Dict[str, Any]] = json.load(f)
        return [JobListing(**item) for item in raw]

    @staticmethod
    def search_jobs(
        jobs: List[JobListing],
        user_skills: List[str],
        query: Optional[str] = None,
        limit: int = 10,
    ) -> List[JobListing]:
        user_skill_tokens: set = set()
        for skill in user_skills:
            user_skill_tokens.update(_normalize_text(skill).split())

        candidates = jobs
        if query:
            q_lower = query.strip().lower()
            candidates = [
                j for j in candidates
                if q_lower in j.title.lower() or q_lower in j.description.lower()
            ]

        scored: List[Tuple[float, JobListing]] = []
        for job in candidates:
            if not job.required_skills:
                pct = 0.0
            else:
                matched = sum(
                    1 for skill in job.required_skills
                    if bool(user_skill_tokens.intersection(set(_normalize_text(skill).split())))
                )
                pct = round(matched / len(job.required_skills) * 100.0, 2)
            scored.append((pct, job))

        scored.sort(key=lambda t: t[0], reverse=True)

        results: List[JobListing] = []
        for pct, job in scored[:limit]:
            results.append(job.copy(update={"match_percentage": pct}))
        return results

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
            upvotes = max(0, (len(gaps) - idx + 1) * 10)
            recs.append(
                RecommendationItem(
                    id=idx,
                    title=f"Intro to {skill}",
                    target_skill=skill,
                    type="Course",
                    duration="4h",
                    upvotes=upvotes,
                )
            )
        return recs

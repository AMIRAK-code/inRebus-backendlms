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
    Advanced Skill & Job Matcher:
    - Normalizes taxonomy data from ESCO/Piemonte formats.
    - Performs case-insensitive role analysis to prevent 404 errors.
    - Ranks jobs using TF-IDF vector similarity for intelligent matching.
    """

    def __init__(self, taxonomy: Dict[str, Any]) -> None:
        self.taxonomy = taxonomy
        self.roles = self._extract_roles(taxonomy)
        
        if not self.roles:
            print("WARNING: Taxonomy is empty. Injecting fallback role.")
            self.roles = {"Fallback Role": {"name": "Fallback Role", "skills": ["system integration"]}}

        # For case-insensitive lookup: map lowercase names to actual keys
        self._role_lookup_map = {name.lower(): name for name in self.roles.keys()}
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
            if not os.path.exists(path):
                print(f"CRITICAL: {path} not found.")
                return cls(taxonomy)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content.startswith("version https://git-lfs"):
                    print(f"CRITICAL: {path} is a Git LFS pointer. Real data not downloaded.")
                elif content:
                    taxonomy = json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading taxonomy {path}: {e}")
            
        return cls(taxonomy)

    @staticmethod
    def load_jobs(path: str) -> List[JobListing]:
        """Safely load job listings, handling Git LFS pointers and empty files."""
        try:
            if not os.path.exists(path):
                print(f"Warning: {path} not found.")
                return []
                
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                
                # Check for Git LFS pointer
                if content.startswith("version https://git-lfs"):
                    print(f"CRITICAL: {path} is a Git LFS pointer. Real data not downloaded.")
                    return []
                
                if not content:
                    return []
                    
                raw: List[Dict[str, Any]] = json.loads(content)
                return [JobListing(**item) for item in raw]
        except Exception as e:
            print(f"Error loading jobs from {path}: {e}")
            return []

    @classmethod
    def from_competenze_csv(cls, path: str) -> "SkillAnalyzer":
        """Parses the specific Regione Piemonte CSV structure."""
        competencies = {}
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    comp_name = row.get("DENOMINAZIONE COMPETENZA", "").strip()
                    tipo = row.get("TIPO", "").strip().lower()
                    desc = row.get("DESCRIZIONE ABILITA'/CONOSCENZA", "").strip()
                    
                    if not comp_name or not desc:
                        continue
                        
                    if comp_name not in competencies:
                        competencies[comp_name] = {"name": comp_name, "skills": [], "knowledge": []}
                        
                    if "abilità" in tipo:
                        competencies[comp_name]["skills"].append(desc)
                    elif "conoscenz" in tipo:
                        competencies[comp_name]["knowledge"].append(desc)
                        
            taxonomy = {"profiles": list(competencies.values())}
        except Exception:
            taxonomy = {"profiles": []}
            
        return cls(taxonomy)

    def get_roles(self) -> List[str]:
        return self.role_names

    def analyze(self, target_role: str, user_input: str) -> AnalysisResult:
        # Case-insensitive lookup logic
        target_norm = target_role.strip().lower()
        if target_norm not in self._role_lookup_map:
            raise KeyError(target_role)
            
        actual_key = self._role_lookup_map[target_norm]

        user_doc = _normalize_text(user_input)
        if not user_doc:
            raise ValueError("Empty input.")

        # Match calculation
        user_vec = self.vectorizer.transform([user_doc])
        role_idx = self.role_names.index(actual_key)
        role_vec = self.role_matrix[role_idx]

        sim = cosine_similarity(user_vec, role_vec)[0][0]
        match_percentage = round(_safe_float(sim) * 100.0, 2)

        role_skills = self._role_skills.get(actual_key, [])
        extracted_skills, gaps = self._compute_skills_and_gaps(user_doc, role_skills)
        recommendations = self._recommend_for_gaps(gaps)

        return AnalysisResult(
            target_role=actual_key,
            extracted_skills=extracted_skills,
            match_percentage=match_percentage,
            skill_gaps=gaps,
            recommendations=recommendations,
        )

    def search_jobs(
        self,
        jobs: List[JobListing],
        user_skills: List[str],
        query: Optional[str] = None,
        limit: int = 10,
    ) -> List[JobListing]:
        """
        Ranks jobs based on semantic similarity between user skills 
        and the job's required skills/description.
        """
        if not jobs:
            return []

        # Create a single document representing the user's validated skill set
        user_profile_doc = _normalize_text(" ".join(user_skills))
        user_vec = self.vectorizer.transform([user_profile_doc])

        scored: List[Tuple[float, JobListing]] = []
        
        # Keyword filter if query is provided
        q_lower = query.strip().lower() if query else None

        for job in jobs:
            # Substring match filter
            if q_lower:
                if q_lower not in job.title.lower() and q_lower not in job.description.lower():
                    continue

            # Calculate match based on TF-IDF similarity of the job requirements
            job_req_doc = _normalize_text(" ".join(job.required_skills) + " " + job.description)
            job_vec = self.vectorizer.transform([job_req_doc])
            
            sim = cosine_similarity(user_vec, job_vec)[0][0]
            pct = round(_safe_float(sim) * 100.0, 2)
            
            # Boost score based on exact skill matches
            user_skill_set = set(_normalize_text(s) for s in user_skills)
            exact_matches = sum(1 for rs in job.required_skills if _normalize_text(rs) in user_skill_set)
            if job.required_skills:
                exact_pct = (exact_matches / len(job.required_skills)) * 100
                # Blend semantic (40%) and exact (60%) scores
                pct = round((pct * 0.4) + (exact_pct * 0.6), 2)

            scored.append((pct, job))

        # Sort by match percentage
        scored.sort(key=lambda t: t[0], reverse=True)

        results: List[JobListing] = []
        for pct, job in scored[:limit]:
            # Use model_copy in Pydantic v2
            cloned_job = job.model_copy(update={"match_percentage": pct})
            results.append(cloned_job)
            
        return results

    def _extract_roles(self, taxonomy: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Handles multiple JSON taxonomy formats (profiles list or roles dict)."""
        out: Dict[str, Dict[str, Any]] = {}
        
        # Format 1: {"profiles": [...]}
        profiles = taxonomy.get("profiles", [])
        if isinstance(profiles, list):
            for p in profiles:
                name = (p.get("name") or "").strip()
                if name:
                    out[name] = p
            if out: return out

        # Format 2: {"roles": {"Title": {...}}}
        roles = taxonomy.get("roles", {})
        if isinstance(roles, dict):
            for name, obj in roles.items():
                if isinstance(obj, dict):
                    out[name.strip()] = obj
            
        return out

    def _extract_skills_list(self, role_obj: Dict[str, Any]) -> List[str]:
        skills = role_obj.get("skills") or []
        out: List[str] = []
        if isinstance(skills, list):
            for s in skills:
                if isinstance(s, str) and s.strip():
                    out.append(s.strip())
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

        desc = role_obj.get("description") or ""
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

            # Partial match logic: if any significant token of the skill is in the CV
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
                    title=f"Learning Path: {skill}",
                    target_skill=skill,
                    type="Course",
                    duration="4h",
                    upvotes=upvotes,
                )
            )
        return recs

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
    industry: Optional[str] = "General"
    metadata: Optional[Dict[str, Any]] = {}
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
    - Normalizes taxonomy data from ESCO, Piemonte, or custom JSON formats.
    - Specifically handles ESCO flat-list relationship structures.
    - Performs case-insensitive role analysis.
    - Ranks jobs using TF-IDF vector similarity blended with exact matches.
    """

    def __init__(self, taxonomy: Any) -> None:
        self.taxonomy = taxonomy
        self.roles = self._extract_roles(taxonomy)
        
        if not self.roles:
            print("WARNING: Taxonomy extraction resulted in 0 roles. Injecting Fallback.")
            self.roles = {"Fallback Role": {"name": "Fallback Role", "skills": ["system integration"]}}
        else:
            print(f"SUCCESS: SkillAnalyzer initialized with {len(self.roles)} unique roles.")

        # For case-insensitive lookup: essential for matching UI input to ESCO labels
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
        """Safely loads JSON, strips Windows BOMs, and provides deep error logging."""
        taxonomy = {"profiles": []}
        try:
            if not os.path.exists(path):
                print(f"CRITICAL: {path} not found.")
                return cls(taxonomy)

            # utf-8-sig automatically strips the invisible Windows BOM if it exists
            with open(path, "r", encoding="utf-8-sig") as f:
                content = f.read().strip()
                
                if not content:
                    print(f"CRITICAL: {path} is completely empty (0 bytes).")
                    return cls(taxonomy)
                    
                if content.startswith("version https://git-lfs"):
                    print(f"CRITICAL: {path} is a Git LFS pointer. Enable GIT_LFS_ENABLED=true in Render.")
                    return cls(taxonomy)
                
                # Load the JSON
                taxonomy = json.loads(content)
                
        except json.JSONDecodeError as e:
            print(f"CRITICAL: JSON Decode Error in {path}: {e}")
            print(f"--- CONTENT PREVIEW (First 200 chars) ---")
            print(repr(content[:200]))
            print(f"-----------------------------------------")
        except Exception as e:
            print(f"Error loading taxonomy {path}: {e}")
            
        return cls(taxonomy)

    @staticmethod
    def load_jobs(path: str) -> List[JobListing]:
        """Safely load job listings, stripping Windows BOMs and handling LFS pointers."""
        try:
            if not os.path.exists(path):
                return []
                
            with open(path, "r", encoding="utf-8-sig") as f:
                content = f.read().strip()
                
                if not content:
                    return []
                    
                if content.startswith("version https://git-lfs"):
                    print(f"CRITICAL: {path} is an LFS pointer.")
                    return []
                    
                raw: List[Dict[str, Any]] = json.loads(content)
                return [JobListing(**item) for item in raw]
                
        except json.JSONDecodeError as e:
            print(f"CRITICAL: JSON Decode Error in {path}: {e}")
            print(f"--- JOBS CONTENT PREVIEW ---")
            print(repr(content[:200]))
            return []
        except Exception as e:
            print(f"Error loading jobs: {e}")
            return []

    def get_roles(self) -> List[str]:
        return self.role_names

    def analyze(self, target_role: str, user_input: str) -> AnalysisResult:
        # Normalize target_role for lookup
        target_norm = target_role.strip().lower()
        if target_norm not in self._role_lookup_map:
            raise KeyError(target_role)
            
        actual_key = self._role_lookup_map[target_norm]
        user_doc = _normalize_text(user_input)
        
        if not user_doc:
            raise ValueError("Empty input.")

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
        if not jobs: return []
        user_profile_doc = _normalize_text(" ".join(user_skills))
        user_vec = self.vectorizer.transform([user_profile_doc])
        scored = []
        q_lower = query.strip().lower() if query else None

        for job in jobs:
            # Substring text search filter
            if q_lower and q_lower not in job.title.lower() and q_lower not in job.description.lower():
                continue
            
            # Semantic matching based on description and required skills
            job_req_doc = _normalize_text(" ".join(job.required_skills) + " " + job.description)
            job_vec = self.vectorizer.transform([job_req_doc])
            sim = cosine_similarity(user_vec, job_vec)[0][0]
            pct = round(_safe_float(sim) * 100.0, 2)
            
            # Exact skill matching blend
            user_skill_set = set(_normalize_text(s) for s in user_skills)
            exact_matches = sum(1 for rs in job.required_skills if _normalize_text(rs) in user_skill_set)
            
            if job.required_skills:
                exact_pct = (exact_matches / len(job.required_skills)) * 100
                pct = round((pct * 0.4) + (exact_pct * 0.6), 2)
                
            scored.append((pct, job))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [job.model_copy(update={"match_percentage": pct}) for pct, job in scored[:limit]]

    def _extract_roles(self, taxonomy: Any) -> Dict[str, Dict[str, Any]]:
        """
        Deep extraction: Aggregates flat ESCO skill relation lists 
        into occupation-centric profile objects.
        """
        out: Dict[str, Dict[str, Any]] = {}
        
        # 1. Handle root-level list (Standard ESCO JSON Export format)
        if isinstance(taxonomy, list):
            for item in taxonomy:
                # Key identifiers for ESCO labels vs generic ones
                name = item.get("occupationLabel") or item.get("name") or item.get("title")
                if name:
                    name = str(name).strip()
                    if name not in out:
                        out[name] = {
                            "name": name, 
                            "skills": [], 
                            "knowledge": [], 
                            "description": item.get("description", "")
                        }
                    
                    # Aggregate skill labels into the parent occupation
                    skill_label = item.get("skillLabel")
                    if skill_label:
                        skill_type = str(item.get("skillType", "")).lower()
                        if "knowledge" in skill_type:
                            out[name]["knowledge"].append(skill_label)
                        else:
                            out[name]["skills"].append(skill_label)
            
            # Final cleaning: deduplicate skills per role
            for role in out:
                out[role]["skills"] = list(set(out[role]["skills"]))
                out[role]["knowledge"] = list(set(out[role]["knowledge"]))
            return out

        # 2. Handle root-level dictionary (Nested formats)
        if isinstance(taxonomy, dict):
            for key in ["profiles", "occupations", "roles"]:
                items = taxonomy.get(key)
                if isinstance(items, list):
                    return self._extract_roles(items)
            
            # Direct dictionary mapping RoleName -> Data
            for key, val in taxonomy.items():
                if isinstance(val, dict) and key not in ["profiles", "metadata", "version"]:
                    role_name = val.get("name") or key
                    out[role_name] = val
                    
        return out

    def _extract_skills_list(self, role_obj: Dict[str, Any]) -> List[str]:
        skills = role_obj.get("skills") or []
        if isinstance(skills, list):
            return sorted(set(str(s).strip() for s in skills if s))
        return []

    def _role_to_document(self, role_obj: Dict[str, Any]) -> str:
        parts = []
        parts.extend(self._extract_skills_list(role_obj))
        knowledge = role_obj.get("knowledge") or []
        if isinstance(knowledge, list):
            parts.extend([str(k).strip() for k in knowledge if k])
        desc = role_obj.get("description") or ""
        if desc: parts.append(str(desc))
        return _normalize_text(" ".join(parts))

    def _compute_skills_and_gaps(self, user_doc: str, role_skills: List[str]) -> Tuple[List[str], List[str]]:
        user_tokens = set(user_doc.split())
        extracted, gaps = [], []
        for skill in role_skills:
            skill_norm = _normalize_text(skill)
            skill_tokens = set(skill_norm.split())
            if skill_tokens and user_tokens.intersection(skill_tokens):
                extracted.append(skill)
            else:
                gaps.append(skill)
        return extracted[:25], gaps[:25]

    def _recommend_for_gaps(self, gaps: List[str]) -> List[RecommendationItem]:
        return [RecommendationItem(
            id=i, title=f"Learning Path: {s}", target_skill=s, 
            type="Course", duration="4h", upvotes=max(0, (len(gaps)-i+1)*10)
        ) for i, s in enumerate(gaps, 1)]

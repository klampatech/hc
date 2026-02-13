import json
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .utils import safe_get, normalize_vectors


@dataclass
class Job:
    """Represents a single job posting."""
    id: str
    title: str
    company: str
    description: str
    apply_url: str

    # Structured fields
    seniority: Optional[str]
    employment_type: Optional[str]
    location: Optional[str]
    is_remote: bool
    salary_min: Optional[float]
    salary_max: Optional[float]
    skills: List[str]

    # Company info
    industry: Optional[str]
    org_type: Optional[str]
    employee_count: Optional[str]
    funding_stage: Optional[str]

    # Embeddings stored as numpy arrays
    embedding_explicit: np.ndarray
    embedding_inferred: np.ndarray
    embedding_company: np.ndarray

    def to_display_string(self) -> str:
        """Format job for display in results."""
        parts = [f"{self.title} - {self.company}"]
        if self.is_remote:
            parts.append("Remote")
        elif self.location:
            parts.append(self.location)
        if self.salary_min and self.salary_max and self.salary_min >= 1000:
            parts.append(f"${self.salary_min/1000:.0f}K-${self.salary_max/1000:.0f}K")
        if self.org_type:
            parts.append(self.org_type.title())
        return " | ".join(parts)


class JobDataset:
    """Loads and indexes the job dataset."""

    def __init__(self, filepath: str):
        self.jobs: List[Job] = []
        self.embeddings_explicit: Optional[np.ndarray] = None
        self.embeddings_inferred: Optional[np.ndarray] = None
        self.embeddings_company: Optional[np.ndarray] = None
        self._load_jobs(filepath)
        self._build_embedding_matrices()

    def _load_jobs(self, filepath: str):
        """Load jobs from JSONL file, line by line for memory efficiency."""
        count = 0
        skipped = 0

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                proc = raw.get("v7_processed_job_data", {}) or {}
                comp = raw.get("v5_processed_company_data", {}) or {}

                # Extract embeddings - skip job if missing
                emb_explicit = proc.get("embedding_explicit_vector")
                emb_inferred = proc.get("embedding_inferred_vector")
                emb_company = proc.get("embedding_company_vector")

                if not emb_explicit or not emb_inferred or not emb_company:
                    skipped += 1
                    continue

                if len(emb_explicit) != 1536 or len(emb_inferred) != 1536 or len(emb_company) != 1536:
                    skipped += 1
                    continue

                job_info = raw.get("job_information", {}) or {}
                v5j = raw.get("v5_processed_job_data", {}) or {}

                # Title: v5 core_job_title > v7 job_titles.explicit.value > job_information.title
                title = (
                    v5j.get("core_job_title")
                    or safe_get(proc, "job_titles", "explicit", "value")
                    or job_info.get("title", "Unknown")
                )

                # Company: v5_company.name > v5_job.company_name > v7 company_profile.name
                company_raw = (
                    comp.get("name")
                    or v5j.get("company_name")
                    or safe_get(proc, "company_profile", "name")
                    or "Unknown"
                )
                company = (
                    company_raw if company_raw not in ("undefined", "null", "None", "")
                    else "Unknown"
                )

                # Seniority: v5_job > v7 experience_requirements
                seniority = (
                    v5j.get("seniority_level")
                    or safe_get(proc, "experience_requirements", "seniority_level")
                )

                # Location: v5 formatted_workplace_location
                location = v5j.get("formatted_workplace_location")

                # Remote: v5 workplace_type or v7 work_arrangement.workplace_type
                workplace_type = (
                    v5j.get("workplace_type")
                    or safe_get(proc, "work_arrangement", "workplace_type")
                    or ""
                )
                is_remote = workplace_type.lower() == "remote" if workplace_type else False

                # Salary: v5 yearly compensation
                salary_min = v5j.get("yearly_min_compensation")
                salary_max = v5j.get("yearly_max_compensation")

                # Skills: v7 skills.explicit[].value > v5 technical_tools
                v7_skills = safe_get(proc, "skills", "explicit")
                if isinstance(v7_skills, list):
                    skills = [s.get("value", "") for s in v7_skills if isinstance(s, dict)]
                else:
                    skills = v5j.get("technical_tools", []) or []

                # Employment type: v5 commitment > v7 work_arrangement.commitment
                commitment = v5j.get("commitment") or safe_get(proc, "work_arrangement", "commitment") or []
                employment_type = commitment[0] if commitment else None

                # Industry: v5_company.industries > v5_job.company_sector_and_industry
                industries = comp.get("industries", []) or []
                industry = (
                    industries[0] if industries
                    else v5j.get("company_sector_and_industry")
                )

                # Org type: v7 company_profile.organization_types > v5_company.is_non_profit
                org_types = safe_get(proc, "company_profile", "organization_types") or []
                is_nonprofit = comp.get("is_non_profit", False)
                if is_nonprofit:
                    org_type = "nonprofit"
                elif org_types:
                    org_type = org_types[0]
                else:
                    org_type = None

                # Employee count: v5_company.num_employees
                employee_count = comp.get("num_employees")

                job = Job(
                    id=raw.get("id", ""),
                    title=title,
                    company=company,
                    description=job_info.get("description", ""),
                    apply_url=raw.get("apply_url", ""),
                    seniority=seniority,
                    employment_type=employment_type,
                    location=location,
                    is_remote=is_remote,
                    salary_min=salary_min,
                    salary_max=salary_max,
                    skills=skills,
                    industry=industry,
                    org_type=org_type,
                    employee_count=str(employee_count) if employee_count else None,
                    funding_stage=comp.get("funding_stage") or comp.get("latest_investment_series"),
                    embedding_explicit=np.array(emb_explicit, dtype=np.float32),
                    embedding_inferred=np.array(emb_inferred, dtype=np.float32),
                    embedding_company=np.array(emb_company, dtype=np.float32),
                )

                self.jobs.append(job)
                count += 1

                if count % 10000 == 0:
                    print(f"  Loaded {count:,} jobs...", file=sys.stderr)

                if count >= 100000:
                    break

        print(f"  Loaded {count:,} jobs ({skipped} skipped)", file=sys.stderr)

    def _build_embedding_matrices(self):
        """Stack embeddings into pre-normalized matrices for fast similarity."""
        if not self.jobs:
            return

        self.embeddings_explicit = normalize_vectors(
            np.vstack([j.embedding_explicit for j in self.jobs])
        )
        self.embeddings_inferred = normalize_vectors(
            np.vstack([j.embedding_inferred for j in self.jobs])
        )
        self.embeddings_company = normalize_vectors(
            np.vstack([j.embedding_company for j in self.jobs])
        )

    def get_job_by_id(self, job_id: str) -> Optional[Job]:
        """Retrieve job by ID."""
        for job in self.jobs:
            if job.id == job_id:
                return job
        return None

    def __len__(self) -> int:
        return len(self.jobs)

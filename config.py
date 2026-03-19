"""
Application configuration and feature flags.

Feature flags are controlled via environment variables.  Set an env var to
"1", "true", "yes", or "on" (case-insensitive) to enable the flag, or to
"0", "false", "no", or "off" to disable it.  Unrecognised values fall back to
the declared default.

Usage (shell / deployment platform)::

    ENABLE_JOB_SEARCH=true uvicorn api:app --host 0.0.0.0 --port 8000

"""
import os


def _bool_env(name: str, default: bool = False) -> bool:
    """Return a boolean value read from an environment variable."""
    val = os.environ.get(name, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

#: Enable the job-search feature (job listings + skill-based job matching).
#: When ``False`` (default), the ``/api/jobs`` endpoints return HTTP 403 so
#: that instances deployed for clients who have not licensed this feature
#: remain unaffected.
ENABLE_JOB_SEARCH: bool = _bool_env("ENABLE_JOB_SEARCH", default=False)

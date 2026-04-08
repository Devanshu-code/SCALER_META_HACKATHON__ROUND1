"""
models.py — Typed Pydantic models for Feed Ranking Environment

Defines the Action, Observation, and State types used by the OpenEnv spec.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# ============================================================
# ACTION — what the agent sends
# ============================================================

class FeedRankingAction(BaseModel):
    """Agent's answer — a ranked list of post IDs (best first)."""
    ranked_post_ids: List[str]


# ============================================================
# OBSERVATION — what the agent sees
# ============================================================

class UserInfo(BaseModel):
    """User profile visible to the agent."""
    id: str
    archetype: str
    stated_interests: Optional[List[str]] = None
    stated_dislikes: Optional[List[str]] = None
    engagement_history: Optional[List[Dict[str, Any]]] = None


class PostInfo(BaseModel):
    """Post features visible to the agent."""
    id: str
    title: str
    topic: str
    content_type: str
    author_id: str
    author_popularity: float
    is_new_creator: bool
    quality_score: float
    is_clickbait: bool
    age_hours: int


class FeedRankingObservation(BaseModel):
    """What the agent receives after reset() or step()."""
    task: Optional[str] = None
    description: Optional[str] = None
    user: Optional[UserInfo] = None
    candidate_posts: Optional[List[PostInfo]] = None
    action_required: Optional[str] = None
    policies: Optional[Dict[str, Any]] = None
    scoring: Optional[str] = None
    message: Optional[str] = None


# ============================================================
# STATE — current environment state
# ============================================================

class FeedRankingState(BaseModel):
    """Current state of the environment."""
    current_task: Optional[str] = None
    current_user_id: Optional[str] = None
    num_candidates: int = 0
    episode_done: bool = False
    steps_taken: int = 0
    total_score: float = 0.0

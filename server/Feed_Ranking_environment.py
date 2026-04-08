"""
Feed_Ranking_environment.py — Environment logic

This file re-exports the environment from app.py for OpenEnv compatibility.
The actual environment logic lives in app.py.
"""

from app import app, FeedRankingEnv, env

__all__ = ["app", "FeedRankingEnv", "env"]

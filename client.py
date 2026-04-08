"""
client.py — OpenEnv client for Feed Ranking Environment

Provides typed client interface for interacting with the environment.
"""

try:
    from openenv import EnvClient
except ImportError:
    EnvClient = object  # fallback if openenv not installed

from models import FeedRankingAction, FeedRankingObservation, FeedRankingState


class FeedRankingEnv(EnvClient):
    """Typed client for the Feed Ranking environment."""

    metadata = {
        "name": "feed-ranking",
        "description": "A social media feed ranking environment for training AI agents.",
    }

    class Action(FeedRankingAction):
        pass

    class Observation(FeedRankingObservation):
        pass

    class State(FeedRankingState):
        pass

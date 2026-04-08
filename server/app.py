
# app.py — Feed Ranking OpenEnv Environment

import random
import sys
import os
from pathlib import Path

# Ensure server/ directory is on the path so 'from data import ...' works
# whether run as 'server.app' from root or 'app' from server/
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from data import (
    POSTS, USERS, TASKS, TOPICS,
    compute_relevance, compute_ndcg, check_policies,
    select_candidates, compute_ideal_ranking,
    select_candidates_enhanced, simulate_engagement
)
import statistics
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# PYDANTIC MODELS — typed inputs
# ============================================================

class RankingAction(BaseModel):
    """Agent's answer — a ranked list of post IDs (best first)."""
    ranked_post_ids: List[str]


# ============================================================
# THE ENVIRONMENT CLASS
# ============================================================

class FeedRankingEnv:
    def __init__(self):
        self.current_task = None
        self.current_user = None
        self.candidate_posts = []
        self.episode_done = False
        self.total_score = 0.0
        self.steps_taken = 0
        self.hard_user_index = 0
        self.hard_scores = []
        self.hard_policy_scores = []
        self.seed = 42

    def reset(self, task: str = "easy"):
        """Start a new episode."""

        if task not in TASKS:
            raise ValueError(f"Unknown task: {task}. Choose from: easy, medium, hard")

        self.current_task = task
        self.episode_done = False
        self.total_score = 0.0
        self.steps_taken = 0

        task_def = TASKS[task]

        if task == "hard":
            self.hard_user_index = 0
            self.hard_scores = []
            self.hard_policy_scores = []
            user_id = task_def["user_ids"][0]
        else:
            user_id = task_def["user_ids"][0]

        self.current_user = USERS[user_id]
        self.candidate_posts = select_candidates_enhanced(
            task_def["num_candidates"],
            seed=self.seed + hash(task) % 1000
        )

        return {
            "observation": self._build_observation(),
            "reward": 0.0,
            "done": False,
            "info": {
                "task": task,
                "num_candidates": len(self.candidate_posts),
                "top_k": task_def["top_k"],
                "steps_taken": 0,
            }
        }

    def step(self, action: dict):
        """Agent submits a ranking — environment scores it."""

        if self.episode_done:
            return {
                "observation": "Episode is done. Call /reset to start again.",
                "reward": 0.0,
                "done": True,
                "info": {"total_score": self.total_score}
            }

        ranked_ids = action.get("ranked_post_ids", [])
        task_def = TASKS[self.current_task]
        top_k = task_def["top_k"]

        # Validate IDs — penalize invalid/destructive actions
        valid_ids = {p["id"] for p in self.candidate_posts}
        invalid = [pid for pid in ranked_ids if pid not in valid_ids]
        if invalid:
            return {
                "observation": f"Invalid post IDs not in candidate pool: {invalid}. Penalty applied.",
                "reward": -0.2,
                "done": False,
                "info": {"error": "invalid_post_ids", "invalid": invalid, "penalty": -0.2}
            }

        # Penalize empty or too-short rankings
        if len(ranked_ids) == 0:
            return {
                "observation": "Empty ranking submitted. Penalty applied.",
                "reward": -0.3,
                "done": False,
                "info": {"error": "empty_ranking", "penalty": -0.3}
            }

        # Penalize duplicate post IDs (lazy/destructive behavior)
        if len(ranked_ids) != len(set(ranked_ids)):
            dupes = [pid for pid in ranked_ids if ranked_ids.count(pid) > 1]
            return {
                "observation": f"Duplicate post IDs detected: {set(dupes)}. Penalty applied.",
                "reward": -0.1,
                "done": False,
                "info": {"error": "duplicate_ids", "duplicates": list(set(dupes)), "penalty": -0.1}
            }

        # Compute NDCG
        ndcg = compute_ndcg(
            self.current_user, ranked_ids, self.candidate_posts, k=top_k
        )

        if self.current_task == "hard":
            policy_result = check_policies(ranked_ids, self.candidate_posts)
            policy_score = policy_result["policy_score"]
            step_score = round(0.6 * ndcg + 0.4 * policy_score, 4)

            self.hard_scores.append(ndcg)
            self.hard_policy_scores.append(policy_score)
            self.hard_user_index += 1

            if self.hard_user_index < len(task_def["user_ids"]):
                next_user_id = task_def["user_ids"][self.hard_user_index]
                self.current_user = USERS[next_user_id]
                self.candidate_posts = select_candidates_enhanced(
                    task_def["num_candidates"],
                    seed=self.seed + hash(next_user_id) % 1000
                )
                self.steps_taken += 1

                return {
                    "observation": self._build_observation(),
                    "reward": step_score,
                    "done": False,
                    "info": {
                        "step_ndcg": ndcg,
                        "step_policy_score": policy_score,
                        "policy_details": policy_result,
                        "step_score": step_score,
                        "users_remaining": len(task_def["user_ids"]) - self.hard_user_index,
                    }
                }
            else:
                avg_ndcg = round(statistics.mean(self.hard_scores), 4)
                avg_policy = round(statistics.mean(self.hard_policy_scores), 4)
                final_score = round(0.6 * avg_ndcg + 0.4 * avg_policy, 4)

                self.total_score = final_score
                self.episode_done = True
                self.steps_taken += 1

                return {
                    "observation": (
                        f"All 3 users scored. Avg NDCG={avg_ndcg}, "
                        f"Avg Policy={avg_policy}, Final={final_score}"
                    ),
                    "reward": final_score,
                    "done": True,
                    "info": {
                        "final_score": final_score,
                        "avg_ndcg": avg_ndcg,
                        "avg_policy_score": avg_policy,
                        "per_user_ndcg": self.hard_scores,
                        "per_user_policy": self.hard_policy_scores,
                    }
                }
        else:
            self.total_score = ndcg
            self.episode_done = True
            self.steps_taken += 1

            if ndcg >= 0.9:
                feedback = f"Excellent ranking! NDCG={ndcg}"
            elif ndcg >= 0.7:
                feedback = f"Good ranking. NDCG={ndcg}"
            elif ndcg >= 0.5:
                feedback = f"Decent ranking. NDCG={ndcg}"
            else:
                feedback = f"Poor ranking. NDCG={ndcg}"

            return {
                "observation": feedback,
                "reward": ndcg,
                "done": True,
                "info": {"score": ndcg, "task": self.current_task}
            }

    def _build_observation(self) -> dict:
        """Build what the agent sees."""

        user = self.current_user
        task_def = TASKS[self.current_task]

        user_info = {"id": user["id"], "archetype": user["archetype"]}
        if user["stated_interests"]:
            user_info["stated_interests"] = user["stated_interests"]
        if user["stated_dislikes"]:
            user_info["stated_dislikes"] = user["stated_dislikes"]
        if user["engagement_history"]:
            user_info["engagement_history"] = user["engagement_history"]

        candidates = []
        for p in self.candidate_posts:
            candidates.append({
                "id": p["id"], "title": p["title"], "topic": p["topic"],
                "content_type": p["content_type"], "author_id": p["author_id"],
                "author_popularity": p["author_popularity"],
                "is_new_creator": p["is_new_creator"],
                "quality_score": p["quality_score"],
                "is_clickbait": p["is_clickbait"], "age_hours": p["age_hours"],
            })

        obs = {
            "task": self.current_task,
            "description": task_def["description"],
            "user": user_info,
            "candidate_posts": candidates,
            "action_required": f"Return top {task_def['top_k']} post IDs ranked best-first",
        }

        if task_def.get("policies_enforced"):
            obs["policies"] = {
                "min_topics_in_top_10": 3,
                "max_same_author_in_top_5": 2,
                "min_new_creator_in_top_10": 1,
                "max_clickbait_in_top_5": 1,
            }
            obs["scoring"] = "0.6 * NDCG + 0.4 * policy_compliance"

        return obs

    def state(self):
        """Return current state."""
        return {
            "current_task": self.current_task,
            "current_user_id": self.current_user["id"] if self.current_user else None,
            "num_candidates": len(self.candidate_posts),
            "episode_done": self.episode_done,
            "steps_taken": self.steps_taken,
            "total_score": self.total_score,
        }


# ============================================================
# FASTAPI SERVER
# ============================================================

app = FastAPI(
    title="Feed Ranking Environment",
    description=(
        "An OpenEnv environment where AI agents learn to rank social media feeds. "
        "Agents must balance user engagement (NDCG), content quality, and platform policies."
    ),
    version="1.0.0"
)

env = FeedRankingEnv()


@app.post("/reset")
def reset(task: str = "easy"):
    return env.reset(task)


@app.post("/step")
def step(action: RankingAction):
    return env.step(action.dict())


@app.get("/state")
def state():
    return env.state()


@app.get("/tasks")
def tasks():
    return {
        task_name: {
            "description": td["description"],
            "action_schema": td["action_schema"],
            "num_candidates": td["num_candidates"],
            "top_k": td["top_k"],
            "num_users": len(td["user_ids"]),
            "policies_enforced": td["policies_enforced"],
        }
        for task_name, td in TASKS.items()
    }


@app.get("/grader")
def grader(task: str = "easy"):
    """Naive baseline — random ranking."""

    if task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")

    task_def = TASKS[task]
    scores = []

    for user_id in task_def["user_ids"]:
        user = USERS[user_id]
        candidates = select_candidates(
            task_def["num_candidates"],
            seed=42 + hash(task if task != "hard" else user_id) % 1000
        )

        rng = random.Random(123)
        naive_ranking = [p["id"] for p in candidates]
        rng.shuffle(naive_ranking)
        naive_ranking = naive_ranking[:task_def["top_k"]]

        ndcg = compute_ndcg(user, naive_ranking, candidates, k=task_def["top_k"])

        if task == "hard":
            policy = check_policies(naive_ranking, candidates)
            score = round(0.6 * ndcg + 0.4 * policy["policy_score"], 4)
        else:
            score = ndcg

        scores.append(score)

    average = round(statistics.mean(scores), 4)

    return {
        "task": task,
        "num_users": len(task_def["user_ids"]),
        "scores": scores,
        "average_score": average,
        "note": "Naive baseline — random ranking of candidate posts"
    }


# ============================================================
# /baseline — AI model against all 3 tasks
# ============================================================

@app.get("/baseline")
def baseline():
    """Run AI model against all 3 tasks."""

    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise HTTPException(status_code=500, detail="No API key found.")

    client = OpenAI(base_url=api_base_url, api_key=api_key)
    all_results = {}

    for task in ["easy", "medium", "hard"]:
        task_def = TASKS[task]
        scores = []
        details = []

        for user_id in task_def["user_ids"]:
            user = USERS[user_id]
            candidates = select_candidates(
                task_def["num_candidates"],
                seed=42 + hash(task if task != "hard" else user_id) % 1000
            )

            prompt = _build_ranking_prompt(user, candidates, task_def)
            ai_response = _call_model(client, model_name, prompt)
            parsed = _parse_ranking_response(ai_response, candidates, task_def["top_k"])

            ndcg = compute_ndcg(user, parsed, candidates, k=task_def["top_k"])

            if task == "hard":
                policy = check_policies(parsed, candidates)
                score = round(0.6 * ndcg + 0.4 * policy["policy_score"], 4)
            else:
                score = ndcg

            scores.append(score)
            details.append({
                "user_id": user_id,
                "parsed_ranking": parsed,
                "ndcg": ndcg,
                "score": score,
            })

        average = round(statistics.mean(scores), 4)
        all_results[task] = {
            "average_score": average,
            "scores": scores,
            "details": details,
        }

    return {
        "model": model_name,
        "results": all_results,
        "summary": {
            "easy": all_results["easy"]["average_score"],
            "medium": all_results["medium"]["average_score"],
            "hard": all_results["hard"]["average_score"],
        }
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _build_ranking_prompt(user: dict, candidates: list, task_def: dict) -> str:
    """Build prompt for AI model."""

    user_info = f"User ID: {user['id']}\nArchetype: {user['archetype']}\n"
    if user.get("stated_interests"):
        user_info += f"Stated interests: {', '.join(user['stated_interests'])}\n"
    if user.get("stated_dislikes"):
        user_info += f"Stated dislikes: {', '.join(user['stated_dislikes'])}\n"
    if user.get("engagement_history"):
        history_str = "\n".join(
            f"  - {h['action']} post about {h['topic']} (post {h['post_id']})"
            for h in user["engagement_history"]
        )
        user_info += f"Engagement history:\n{history_str}\n"

    posts_str = "\n".join(
        f"  {p['id']}: \"{p['title']}\" | topic={p['topic']} | type={p['content_type']} | "
        f"quality={p['quality_score']} | clickbait={p['is_clickbait']} | "
        f"new_creator={p['is_new_creator']} | age={p['age_hours']}h | "
        f"author={p['author_id']} (pop={p['author_popularity']})"
        for p in candidates
    )

    policy_str = ""
    if task_def.get("policies_enforced"):
        policy_str = (
            "\n\nPLATFORM POLICIES (must satisfy):\n"
            "- At least 3 different topics in your top 10\n"
            "- Max 2 posts from same author in top 5\n"
            "- At least 1 new creator post in top 10\n"
            "- Max 1 clickbait post in top 5\n"
        )

    top_k = task_def["top_k"]

    return f"""You are a feed ranking algorithm. Rank the top {top_k} posts for this user.

USER PROFILE:
{user_info}

CANDIDATE POSTS:
{posts_str}
{policy_str}

Respond with ONLY a JSON object: {{"ranked_post_ids": ["p01", "p39", ...]}}
Exactly {top_k} post IDs, ordered best-first. No explanation."""


def _call_model(client: OpenAI, model_name: str, prompt: str) -> str:
    """Call any OpenAI-compatible API."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        content = response.choices[0].message.content
        return content.strip() if content else "error: empty response"
    except Exception as e:
        print(f"API error: {str(e)}")
        return f"error: {str(e)}"


def _parse_ranking_response(response: str, candidates: list, top_k: int) -> list:
    """Parse AI response into list of post IDs."""

    valid_ids = {p["id"] for p in candidates}

    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(response[start:end])
            ids = parsed.get("ranked_post_ids", [])
            ids = [pid for pid in ids if pid in valid_ids]
            if ids:
                return ids[:top_k]
    except:
        pass

    rng = random.Random(999)
    fallback = [p["id"] for p in candidates]
    rng.shuffle(fallback)
    return fallback[:top_k]


# ============================================================
#  3 ENDPOINTS (required by OpenEnv validator): /metadata, /schema, /mcp
# ============================================================


# --- 1. /metadata endpoint ---
@app.get("/metadata")
def metadata():
    """Return environment name and description (required by OpenEnv validator)."""
    return {
        "name": "feed-ranking",
        "description": (
            "A social media feed ranking environment where AI agents learn to "
            "personalize content feeds. Agents must balance user engagement (NDCG), "
            "content quality, and platform policies across three difficulty levels."
        ),
        "version": "1.0.0",
        "author": "DevanshuDon",
        "tasks": ["easy", "medium", "hard"],
    }


# --- 2. /schema endpoint ---
@app.get("/schema")
def schema():
    """Return action, observation, and state schemas (required by OpenEnv validator)."""
    return {
        "action": {
            "type": "object",
            "properties": {
                "ranked_post_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 10 post IDs in ranked order (best first)",
                }
            },
            "required": ["ranked_post_ids"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Current task name"},
                "description": {"type": "string", "description": "Task description"},
                "user": {
                    "type": "object",
                    "description": "User profile with id, archetype, interests, and history",
                },
                "candidate_posts": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of candidate posts to rank",
                },
                "action_required": {"type": "string", "description": "What the agent must return"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "current_task": {"type": "string"},
                "current_user_id": {"type": "string"},
                "num_candidates": {"type": "integer"},
                "episode_done": {"type": "boolean"},
                "steps_taken": {"type": "integer"},
                "total_score": {"type": "number"},
            },
        },
    }


# --- 3. /mcp endpoint (JSON-RPC) ---
@app.post("/mcp")
async def mcp_endpoint(request_body: dict = {}):
    """MCP JSON-RPC endpoint (required by OpenEnv validator)."""

    method = request_body.get("method", "")
    req_id = request_body.get("id", 1)

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "feed-ranking",
                    "version": "1.0.0",
                },
                "capabilities": {
                    "tools": {"listChanged": False},
                },
            },
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "reset",
                        "description": "Start a new episode with a task (easy/medium/hard)",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string", "enum": ["easy", "medium", "hard"]}
                            },
                        },
                    },
                    {
                        "name": "step",
                        "description": "Submit a ranked list of post IDs",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "ranked_post_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                }
                            },
                            "required": ["ranked_post_ids"],
                        },
                    },
                    {
                        "name": "state",
                        "description": "Get the current environment state",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ],
            },
        }

    # Default response for any other method
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {},
    }





@app.get("/health")
def health():
    return {"status": "healthy"}

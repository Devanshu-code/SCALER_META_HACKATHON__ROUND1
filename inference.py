

import os
import json
import random
import statistics
from typing import List, Optional

from openai import OpenAI

from server.data import (
    POSTS, USERS, TASKS,
    compute_ndcg, check_policies, select_candidates
)

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# CONFIGURATION — read from environment variables
# ============================================================

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = "feed-ranking"
TEMPERATURE = 0.1
MAX_TOKENS = 200


# ============================================================
# STRUCTURED STDOUT LOGGING — required by hackathon
# ============================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ============================================================
# PROMPT BUILDING
# ============================================================

def build_ranking_prompt(user: dict, candidates: list, task_def: dict) -> str:
    """Build prompt for the AI model to rank posts."""

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


# ============================================================
# MODEL INTERACTION
# ============================================================

def call_model(client: OpenAI, prompt: str) -> str:
    """Call any OpenAI-compatible API. Handles errors gracefully."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = completion.choices[0].message.content or ""
        return response_text.strip()
    except Exception as exc:
        return ""


# ============================================================
# RESPONSE PARSING
# ============================================================

def parse_ranking_response(response: str, candidates: list, top_k: int) -> list:
    """Parse AI response into list of post IDs."""
    valid_ids = {p["id"] for p in candidates}

    if response:
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(response[start:end])
                ids = parsed.get("ranked_post_ids", [])
                ids = [pid for pid in ids if pid in valid_ids]
                if ids:
                    return ids[:top_k]
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: random ordering
    rng = random.Random(999)
    fallback = [p["id"] for p in candidates]
    rng.shuffle(fallback)
    return fallback[:top_k]


# ============================================================
# MAIN — Run baseline inference on all 3 tasks
# ============================================================

def main() -> None:
    """Run baseline inference on all 3 tasks with structured stdout."""

    if not API_KEY:
        print("[END] success=false steps=0 score=0.000 rewards=", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in ["easy", "medium", "hard"]:
        task_def = TASKS[task]
        rewards = []
        step_count = 0
        error_msg = None

        log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

        try:
            for user_id in task_def["user_ids"]:
                user = USERS[user_id]
                candidates = select_candidates(
                    task_def["num_candidates"],
                    seed=42 + hash(task if task != "hard" else user_id) % 1000
                )

                # Build prompt and get AI response
                prompt = build_ranking_prompt(user, candidates, task_def)
                ai_response = call_model(client, prompt)

                # Parse response into ranking
                parsed = parse_ranking_response(ai_response, candidates, task_def["top_k"])

                # Score the ranking
                ndcg = compute_ndcg(user, parsed, candidates, k=task_def["top_k"])

                if task == "hard":
                    policy = check_policies(parsed, candidates)
                    score = round(0.6 * ndcg + 0.4 * policy["policy_score"], 4)
                else:
                    score = ndcg

                rewards.append(score)
                step_count += 1

                # Determine if this is the last step
                is_last = (user_id == task_def["user_ids"][-1])
                action_str = f"rank({user_id})"

                log_step(
                    step=step_count,
                    action=action_str,
                    reward=score,
                    done=is_last,
                    error=None,
                )

            # Compute final score for this task
            final_score = round(statistics.mean(rewards), 4) if rewards else 0.0
            final_score = min(max(final_score, 0.0), 1.0)
            success = final_score > 0.0

        except Exception as exc:
            error_msg = str(exc)
            final_score = 0.0
            success = False

        log_end(
            success=success,
            steps=step_count,
            score=final_score,
            rewards=rewards,
        )


if __name__ == "__main__":
    main()
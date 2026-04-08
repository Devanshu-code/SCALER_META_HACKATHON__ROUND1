---
title: Feed Ranking
emoji: 🐢
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
license: mit
app_port: 7860
tags:
  - openenv
---

# Feed Ranking Environment

A social media feed ranking environment where AI agents learn to personalize content feeds. Agents must balance user engagement, content quality, and platform policies — simulating the real-world challenge faced by recommendation systems at companies like Instagram, YouTube, and Twitter.

## Why Feed Ranking?

Every time you open a social media app, a ranking algorithm decides what you see. This environment simulates that exact problem: given a user profile and a pool of candidate posts, rank the feed to maximize engagement while respecting platform policies. It features Bayesian-inspired user simulation with hidden preference weights, multiple user archetypes, dynamic post generation, stochastic engagement noise, and realistic content moderation constraints.

## Tasks

### Task 1: Easy — Explicit Preferences
- **Difficulty:** Easy
- **Description:** Rank 15 candidate posts for a single user who clearly states their interests and dislikes
- **Challenge:** Match posts to stated preferences, prioritize quality over clickbait
- **Scoring:** NDCG@10 (0.0 to 1.0)

### Task 2: Medium — Implicit Preferences
- **Difficulty:** Medium
- **Description:** Rank 25 candidate posts for a user with NO stated preferences — only engagement history
- **Challenge:** Infer user interests from past behavior (clicks, likes, skips), handle ambiguous signals
- **Scoring:** NDCG@10 (0.0 to 1.0)

### Task 3: Hard — Multi-User with Policy Constraints
- **Difficulty:** Hard
- **Description:** Rank 20 posts for 3 different user archetypes sequentially while satisfying platform policies
- **Challenge:** Balance personalization across diverse users AND enforce content policies simultaneously
- **Policies enforced:**
  - Minimum 3 different topics in top 10
  - Max 2 posts from same author in top 5
  - At least 1 new creator post in top 10
  - Max 1 clickbait post in top 5
- **Scoring:** 0.6 × average_NDCG + 0.4 × policy_compliance (0.0 to 1.0)

## Observation Space

When the agent calls `/reset`, it receives:

```json
{
  "observation": {
    "task": "easy",
    "description": "Rank 15 candidate posts for a single user...",
    "user": {
      "id": "user_easy",
      "archetype": "casual_scroller",
      "stated_interests": ["tech", "gaming"],
      "stated_dislikes": ["fashion", "politics"],
      "engagement_history": []
    },
    "candidate_posts": [
      {
        "id": "p01",
        "title": "New AI chip breaks speed records",
        "topic": "tech",
        "content_type": "text",
        "author_id": "a01",
        "author_popularity": 0.85,
        "is_new_creator": false,
        "quality_score": 0.9,
        "is_clickbait": false,
        "age_hours": 2
      }
    ],
    "action_required": "Return top 10 post IDs ranked best-first"
  },
  "reward": 0.0,
  "done": false
}
```

## Action Space

The agent submits a ranked list of post IDs:

```json
{
  "ranked_post_ids": ["p39", "p44", "p42", "p40", "p07", "p32", "p20", "p12", "p47", "p25"]
}
```

## Reward Function

The reward function provides continuous, meaningful signal with penalties for undesirable behavior:

**Positive signals:**
- **NDCG (Normalized Discounted Cumulative Gain):** Measures ranking quality against the ideal ordering based on hidden user preferences. Rewards partial credit — a slightly wrong ordering still scores better than random.
- **Policy Compliance (Hard task only):** Fraction of platform policies satisfied (0.0 to 1.0). Combined with NDCG as: `0.6 × NDCG + 0.4 × policy_score`.

**Engagement Model Factors:**
- Topic preference weight (from hidden user weights)
- Content type multiplier (video/image/text preference)
- Quality bonus (higher quality posts boost engagement)
- Freshness bonus (newer posts get a small boost)
- New creator discovery bonus

**Penalties for undesirable behavior:**
- **-0.2** for submitting post IDs not in the candidate pool
- **-0.3** for submitting an empty ranking
- **-0.1** for submitting duplicate post IDs
- **Clickbait penalty** applied during relevance scoring (critical users penalize heavily)

## Dynamic Post Generation

Each episode mixes static posts (60%) with dynamically generated posts (40%). Dynamic posts have randomized titles, quality scores, clickbait probability, and author assignments. This prevents agents from memorizing answers and ensures genuine generalization.

## Stochastic User Simulation

The engagement model includes bounded Gaussian noise (σ=0.05) on relevance scores, simulating real-world user unpredictability. The same post won't always receive the exact same engagement — just like real users whose behavior varies with mood, attention, and context.

## User Archetypes

| Archetype | Behavior |
|-----------|----------|
| Casual Scroller | Clicks on trending content, prefers video, moderate quality tolerance |
| Niche Enthusiast | Strong topic preferences, engages deeply with specific interests |
| Critical User | Penalizes clickbait heavily, values quality and authenticity |

## Baseline Scores

### Naive Baseline (Random Ranking)
| Task | Score |
|------|-------|
| Easy | 0.6465 |
| Medium | 0.7233 |
| Hard | 0.7235 |

### AI Baseline (Nemotron 3 Super — 120B)
| Task | Score |
|------|-------|
| Easy | 0.8930 |
| Medium | 0.8020 |
| Hard | 0.7360 |
| **Overall** | **0.8103** |

The AI baseline significantly outperforms naive random ranking, demonstrating that the environment creates a meaningful learning signal. The difficulty progression is clear: easy (explicit preferences) scores highest, medium (implicit inference) is harder, and hard (multi-user + policies) is the most challenging.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset?task=easy\|medium\|hard` | POST | Start a new episode |
| `/step` | POST | Submit ranking, get score |
| `/state` | GET | Current environment state |
| `/tasks` | GET | List all tasks with schemas |
| `/grader?task=easy\|medium\|hard` | GET | Naive baseline scores |
| `/baseline` | GET | AI model baseline scores |
| `/metadata` | GET | Environment name and description |
| `/schema` | GET | Action, observation, state schemas |
| `/mcp` | POST | MCP JSON-RPC endpoint |
| `/health` | GET | Health check |

## Setup & Usage

### Local Development

```bash
# Clone the repository
git clone https://huggingface.co/spaces/DevanshuDon/feed-ranking
cd feed-ranking

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app:app --reload

# Open API docs
# http://127.0.0.1:8000/docs
```

### Run Baseline Inference

```bash
# Set environment variables
export API_BASE_URL=https://openrouter.ai/api/v1
export MODEL_NAME=nvidia/nemotron-3-super-120b-a12b:free
export HF_TOKEN=your-api-key

# Run inference
python inference.py
```

Expected output format:
```
[START] task=easy env=feed-ranking model=nvidia/nemotron-3-super-120b-a12b:free
[STEP] step=1 action=rank(user_easy) reward=0.89 done=true error=null
[END] success=true steps=1 score=0.893 rewards=0.89
```

### Docker

```bash
docker build -t feed-ranking .
docker run -p 7860:7860 feed-ranking
```

## Technical Details

- **Framework:** FastAPI
- **Language:** Python 3.9
- **Scoring:** NDCG with Bayesian-inspired engagement simulation
- **Posts:** 50 static + dynamically generated posts across 8 topics
- **Users:** 5 simulated user profiles with hidden preference weights
- **Noise:** Gaussian engagement noise for realistic user simulation
- **Penalties:** Negative rewards for invalid, empty, or duplicate actions
- **Deployment:** HuggingFace Spaces (Docker)
- **OpenEnv Validator:** 6/6 checks passed

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | API endpoint for the LLM |
| `MODEL_NAME` | Model identifier for inference |
| `HF_TOKEN` | API key for authentication |

## Author

**DevanshuDon** — Built for the OpenEnv Hackathon 2026

"""
data.py — Feed Ranking Environment Data

Contains:
- 50 social media posts with rich features
- Simulated user profiles with hidden preference weights
- Engagement probability model (Bayesian-inspired)
- Task definitions for easy / medium / hard
"""

import random
import math
import time

# ============================================================
# TOPICS & CONTENT TYPES
# ============================================================

TOPICS = ["tech", "cooking", "fitness", "fashion", "travel", "politics", "gaming", "music"]

CONTENT_TYPES = ["image", "video", "text"]

# ============================================================
# POST POOL — 50 diverse social media posts
# ============================================================

POSTS = [
    # ── Tech posts (8) ──
    {"id": "p01", "title": "New AI chip breaks speed records", "topic": "tech", "content_type": "text",
     "author_id": "a01", "author_popularity": 0.85, "is_new_creator": False, "quality_score": 0.9, "is_clickbait": False, "age_hours": 2},
    {"id": "p02", "title": "10 Python tricks you NEED to know!!!", "topic": "tech", "content_type": "video",
     "author_id": "a02", "author_popularity": 0.7, "is_new_creator": False, "quality_score": 0.4, "is_clickbait": True, "age_hours": 5},
    {"id": "p03", "title": "Building a home server with Raspberry Pi", "topic": "tech", "content_type": "video",
     "author_id": "a03", "author_popularity": 0.5, "is_new_creator": True, "quality_score": 0.8, "is_clickbait": False, "age_hours": 12},
    {"id": "p04", "title": "React vs Vue in 2026 — honest comparison", "topic": "tech", "content_type": "text",
     "author_id": "a04", "author_popularity": 0.65, "is_new_creator": False, "quality_score": 0.85, "is_clickbait": False, "age_hours": 8},
    {"id": "p05", "title": "This AI tool will DESTROY your job!!!", "topic": "tech", "content_type": "video",
     "author_id": "a05", "author_popularity": 0.9, "is_new_creator": False, "quality_score": 0.3, "is_clickbait": True, "age_hours": 1},
    {"id": "p06", "title": "Understanding transformer architectures", "topic": "tech", "content_type": "text",
     "author_id": "a06", "author_popularity": 0.4, "is_new_creator": True, "quality_score": 0.95, "is_clickbait": False, "age_hours": 24},
    {"id": "p07", "title": "My mechanical keyboard collection", "topic": "tech", "content_type": "image",
     "author_id": "a07", "author_popularity": 0.55, "is_new_creator": False, "quality_score": 0.6, "is_clickbait": False, "age_hours": 18},
    {"id": "p08", "title": "Linux vs Windows — the FINAL answer", "topic": "tech", "content_type": "video",
     "author_id": "a02", "author_popularity": 0.7, "is_new_creator": False, "quality_score": 0.5, "is_clickbait": True, "age_hours": 3},

    # ── Cooking posts (7) ──
    {"id": "p09", "title": "Authentic ramen from scratch", "topic": "cooking", "content_type": "video",
     "author_id": "a08", "author_popularity": 0.75, "is_new_creator": False, "quality_score": 0.9, "is_clickbait": False, "age_hours": 6},
    {"id": "p10", "title": "5-minute breakfast ideas for busy mornings", "topic": "cooking", "content_type": "image",
     "author_id": "a09", "author_popularity": 0.6, "is_new_creator": True, "quality_score": 0.7, "is_clickbait": False, "age_hours": 10},
    {"id": "p11", "title": "You WON'T BELIEVE this cake hack!!!", "topic": "cooking", "content_type": "video",
     "author_id": "a10", "author_popularity": 0.8, "is_new_creator": False, "quality_score": 0.35, "is_clickbait": True, "age_hours": 4},
    {"id": "p12", "title": "Fermentation basics — a beginner guide", "topic": "cooking", "content_type": "text",
     "author_id": "a11", "author_popularity": 0.45, "is_new_creator": True, "quality_score": 0.85, "is_clickbait": False, "age_hours": 48},
    {"id": "p13", "title": "Street food tour of Bangkok", "topic": "cooking", "content_type": "video",
     "author_id": "a12", "author_popularity": 0.7, "is_new_creator": False, "quality_score": 0.8, "is_clickbait": False, "age_hours": 15},
    {"id": "p14", "title": "Meal prep Sunday — full week in 2 hours", "topic": "cooking", "content_type": "image",
     "author_id": "a09", "author_popularity": 0.6, "is_new_creator": True, "quality_score": 0.75, "is_clickbait": False, "age_hours": 20},
    {"id": "p15", "title": "Secret ingredient that changes everything", "topic": "cooking", "content_type": "text",
     "author_id": "a10", "author_popularity": 0.8, "is_new_creator": False, "quality_score": 0.4, "is_clickbait": True, "age_hours": 7},

    # ── Fitness posts (6) ──
    {"id": "p16", "title": "30-day pushup challenge results", "topic": "fitness", "content_type": "video",
     "author_id": "a13", "author_popularity": 0.65, "is_new_creator": False, "quality_score": 0.75, "is_clickbait": False, "age_hours": 9},
    {"id": "p17", "title": "Why your diet is KILLING you!!!", "topic": "fitness", "content_type": "text",
     "author_id": "a14", "author_popularity": 0.85, "is_new_creator": False, "quality_score": 0.25, "is_clickbait": True, "age_hours": 2},
    {"id": "p18", "title": "Yoga flow for back pain relief", "topic": "fitness", "content_type": "video",
     "author_id": "a15", "author_popularity": 0.5, "is_new_creator": True, "quality_score": 0.9, "is_clickbait": False, "age_hours": 14},
    {"id": "p19", "title": "Home gym setup under $500", "topic": "fitness", "content_type": "image",
     "author_id": "a16", "author_popularity": 0.55, "is_new_creator": False, "quality_score": 0.7, "is_clickbait": False, "age_hours": 30},
    {"id": "p20", "title": "Marathon training plan — week 1 to race day", "topic": "fitness", "content_type": "text",
     "author_id": "a13", "author_popularity": 0.65, "is_new_creator": False, "quality_score": 0.85, "is_clickbait": False, "age_hours": 22},
    {"id": "p21", "title": "Get abs in 7 days — NO equipment!!!", "topic": "fitness", "content_type": "video",
     "author_id": "a14", "author_popularity": 0.85, "is_new_creator": False, "quality_score": 0.2, "is_clickbait": True, "age_hours": 1},

    # ── Fashion posts (6) ──
    {"id": "p22", "title": "Capsule wardrobe guide for minimalists", "topic": "fashion", "content_type": "image",
     "author_id": "a17", "author_popularity": 0.6, "is_new_creator": False, "quality_score": 0.8, "is_clickbait": False, "age_hours": 16},
    {"id": "p23", "title": "Thrift flip — $5 jacket transformation", "topic": "fashion", "content_type": "video",
     "author_id": "a18", "author_popularity": 0.7, "is_new_creator": True, "quality_score": 0.75, "is_clickbait": False, "age_hours": 11},
    {"id": "p24", "title": "Celebrities HATE this fashion trick!!!", "topic": "fashion", "content_type": "text",
     "author_id": "a19", "author_popularity": 0.9, "is_new_creator": False, "quality_score": 0.2, "is_clickbait": True, "age_hours": 3},
    {"id": "p25", "title": "Spring 2026 color trends", "topic": "fashion", "content_type": "image",
     "author_id": "a17", "author_popularity": 0.6, "is_new_creator": False, "quality_score": 0.7, "is_clickbait": False, "age_hours": 28},
    {"id": "p26", "title": "Sustainable fashion brands to support", "topic": "fashion", "content_type": "text",
     "author_id": "a20", "author_popularity": 0.45, "is_new_creator": True, "quality_score": 0.85, "is_clickbait": False, "age_hours": 36},
    {"id": "p27", "title": "Sneaker collection tour 2026", "topic": "fashion", "content_type": "video",
     "author_id": "a18", "author_popularity": 0.7, "is_new_creator": True, "quality_score": 0.65, "is_clickbait": False, "age_hours": 8},

    # ── Travel posts (6) ──
    {"id": "p28", "title": "Hidden gems in Portugal nobody talks about", "topic": "travel", "content_type": "video",
     "author_id": "a21", "author_popularity": 0.7, "is_new_creator": False, "quality_score": 0.85, "is_clickbait": False, "age_hours": 13},
    {"id": "p29", "title": "Budget backpacking Southeast Asia", "topic": "travel", "content_type": "text",
     "author_id": "a22", "author_popularity": 0.5, "is_new_creator": True, "quality_score": 0.8, "is_clickbait": False, "age_hours": 25},
    {"id": "p30", "title": "This country will SHOCK you!!!", "topic": "travel", "content_type": "video",
     "author_id": "a23", "author_popularity": 0.85, "is_new_creator": False, "quality_score": 0.3, "is_clickbait": True, "age_hours": 2},
    {"id": "p31", "title": "Solo travel safety tips for women", "topic": "travel", "content_type": "text",
     "author_id": "a24", "author_popularity": 0.55, "is_new_creator": False, "quality_score": 0.9, "is_clickbait": False, "age_hours": 40},
    {"id": "p32", "title": "Japanese train system explained", "topic": "travel", "content_type": "video",
     "author_id": "a21", "author_popularity": 0.7, "is_new_creator": False, "quality_score": 0.75, "is_clickbait": False, "age_hours": 19},
    {"id": "p33", "title": "Van life — one year update", "topic": "travel", "content_type": "image",
     "author_id": "a25", "author_popularity": 0.6, "is_new_creator": True, "quality_score": 0.7, "is_clickbait": False, "age_hours": 10},

    # ── Politics posts (5) ──
    {"id": "p34", "title": "New climate policy explained simply", "topic": "politics", "content_type": "text",
     "author_id": "a26", "author_popularity": 0.6, "is_new_creator": False, "quality_score": 0.85, "is_clickbait": False, "age_hours": 5},
    {"id": "p35", "title": "Election results breakdown by state", "topic": "politics", "content_type": "image",
     "author_id": "a27", "author_popularity": 0.75, "is_new_creator": False, "quality_score": 0.8, "is_clickbait": False, "age_hours": 8},
    {"id": "p36", "title": "Politicians DON'T want you to see this!!!", "topic": "politics", "content_type": "video",
     "author_id": "a28", "author_popularity": 0.9, "is_new_creator": False, "quality_score": 0.15, "is_clickbait": True, "age_hours": 1},
    {"id": "p37", "title": "How local government actually works", "topic": "politics", "content_type": "video",
     "author_id": "a29", "author_popularity": 0.35, "is_new_creator": True, "quality_score": 0.9, "is_clickbait": False, "age_hours": 50},
    {"id": "p38", "title": "Tax changes for 2026 — what you need to know", "topic": "politics", "content_type": "text",
     "author_id": "a26", "author_popularity": 0.6, "is_new_creator": False, "quality_score": 0.75, "is_clickbait": False, "age_hours": 15},

    # ── Gaming posts (6) ──
    {"id": "p39", "title": "Elden Ring DLC — honest review", "topic": "gaming", "content_type": "video",
     "author_id": "a30", "author_popularity": 0.8, "is_new_creator": False, "quality_score": 0.85, "is_clickbait": False, "age_hours": 7},
    {"id": "p40", "title": "Best budget gaming setup 2026", "topic": "gaming", "content_type": "image",
     "author_id": "a31", "author_popularity": 0.55, "is_new_creator": True, "quality_score": 0.7, "is_clickbait": False, "age_hours": 20},
    {"id": "p41", "title": "This game will RUIN your life!!!", "topic": "gaming", "content_type": "video",
     "author_id": "a32", "author_popularity": 0.85, "is_new_creator": False, "quality_score": 0.25, "is_clickbait": True, "age_hours": 3},
    {"id": "p42", "title": "Speedrunning history — a documentary", "topic": "gaming", "content_type": "video",
     "author_id": "a33", "author_popularity": 0.5, "is_new_creator": False, "quality_score": 0.9, "is_clickbait": False, "age_hours": 45},
    {"id": "p43", "title": "Indie games you missed in 2025", "topic": "gaming", "content_type": "text",
     "author_id": "a31", "author_popularity": 0.55, "is_new_creator": True, "quality_score": 0.8, "is_clickbait": False, "age_hours": 30},
    {"id": "p44", "title": "Competitive Valorant tips from a pro", "topic": "gaming", "content_type": "video",
     "author_id": "a30", "author_popularity": 0.8, "is_new_creator": False, "quality_score": 0.75, "is_clickbait": False, "age_hours": 12},

    # ── Music posts (6) ──
    {"id": "p45", "title": "How to produce lo-fi beats at home", "topic": "music", "content_type": "video",
     "author_id": "a34", "author_popularity": 0.65, "is_new_creator": False, "quality_score": 0.8, "is_clickbait": False, "age_hours": 9},
    {"id": "p46", "title": "Vinyl collection — my top 20 albums", "topic": "music", "content_type": "image",
     "author_id": "a35", "author_popularity": 0.5, "is_new_creator": True, "quality_score": 0.7, "is_clickbait": False, "age_hours": 22},
    {"id": "p47", "title": "This song will make you CRY!!!", "topic": "music", "content_type": "video",
     "author_id": "a36", "author_popularity": 0.9, "is_new_creator": False, "quality_score": 0.3, "is_clickbait": True, "age_hours": 2},
    {"id": "p48", "title": "Music theory basics — scales and chords", "topic": "music", "content_type": "text",
     "author_id": "a37", "author_popularity": 0.4, "is_new_creator": True, "quality_score": 0.9, "is_clickbait": False, "age_hours": 35},
    {"id": "p49", "title": "Live concert photography tips", "topic": "music", "content_type": "image",
     "author_id": "a34", "author_popularity": 0.65, "is_new_creator": False, "quality_score": 0.65, "is_clickbait": False, "age_hours": 16},
    {"id": "p50", "title": "Guitar vs piano — which to learn first", "topic": "music", "content_type": "video",
     "author_id": "a38", "author_popularity": 0.55, "is_new_creator": False, "quality_score": 0.75, "is_clickbait": False, "age_hours": 11},
]


# ============================================================
# USER PROFILES — with hidden preference weights
# ============================================================

USERS = {
    # ── User for EASY task: explicit preferences clearly stated ──
    "user_easy": {
        "id": "user_easy",
        "archetype": "casual_scroller",
        "stated_interests": ["tech", "gaming"],        # VISIBLE to agent
        "stated_dislikes": ["fashion", "politics"],     # VISIBLE to agent
        "hidden_weights": {                              # NOT visible — used for scoring
            "tech": 0.9, "cooking": 0.3, "fitness": 0.2,
            "fashion": 0.05, "travel": 0.4, "politics": 0.05,
            "gaming": 0.85, "music": 0.5
        },
        "content_type_preference": {"video": 1.2, "image": 1.0, "text": 0.8},
        "engagement_history": [],   # empty for easy — interests are stated
    },

    # ── User for MEDIUM task: preferences must be inferred from history ──
    "user_medium": {
        "id": "user_medium",
        "archetype": "niche_enthusiast",
        "stated_interests": [],                          # NOT given — agent must infer
        "stated_dislikes": [],                           # NOT given
        "hidden_weights": {
            "tech": 0.3, "cooking": 0.9, "fitness": 0.7,
            "fashion": 0.1, "travel": 0.6, "politics": 0.1,
            "gaming": 0.2, "music": 0.4
        },
        "content_type_preference": {"video": 1.1, "image": 1.0, "text": 0.9},
        "engagement_history": [
            # past interactions the agent CAN see — must infer preferences from these
            {"post_id": "p09", "action": "liked",   "topic": "cooking"},
            {"post_id": "p12", "action": "liked",   "topic": "cooking"},
            {"post_id": "p13", "action": "clicked", "topic": "cooking"},
            {"post_id": "p16", "action": "liked",   "topic": "fitness"},
            {"post_id": "p18", "action": "clicked", "topic": "fitness"},
            {"post_id": "p20", "action": "liked",   "topic": "fitness"},
            {"post_id": "p28", "action": "clicked", "topic": "travel"},
            {"post_id": "p29", "action": "liked",   "topic": "travel"},
            {"post_id": "p22", "action": "skipped", "topic": "fashion"},
            {"post_id": "p24", "action": "skipped", "topic": "fashion"},
            {"post_id": "p36", "action": "skipped", "topic": "politics"},
            {"post_id": "p41", "action": "skipped", "topic": "gaming"},
            {"post_id": "p01", "action": "clicked", "topic": "tech"},
            {"post_id": "p47", "action": "skipped", "topic": "music"},
            {"post_id": "p45", "action": "clicked", "topic": "music"},
        ],
    },

    # ── Users for HARD task: 3 different archetypes + policy constraints ──
    "user_hard_1": {
        "id": "user_hard_1",
        "archetype": "casual_scroller",
        "stated_interests": ["travel", "cooking"],
        "stated_dislikes": [],
        "hidden_weights": {
            "tech": 0.3, "cooking": 0.8, "fitness": 0.4,
            "fashion": 0.5, "travel": 0.9, "politics": 0.2,
            "gaming": 0.1, "music": 0.6
        },
        "content_type_preference": {"video": 1.3, "image": 1.0, "text": 0.7},
        "engagement_history": [
            {"post_id": "p28", "action": "liked",   "topic": "travel"},
            {"post_id": "p09", "action": "liked",   "topic": "cooking"},
            {"post_id": "p33", "action": "clicked", "topic": "travel"},
            {"post_id": "p45", "action": "clicked", "topic": "music"},
        ],
    },
    "user_hard_2": {
        "id": "user_hard_2",
        "archetype": "niche_enthusiast",
        "stated_interests": [],
        "stated_dislikes": [],
        "hidden_weights": {
            "tech": 0.95, "cooking": 0.1, "fitness": 0.15,
            "fashion": 0.05, "travel": 0.2, "politics": 0.3,
            "gaming": 0.8, "music": 0.3
        },
        "content_type_preference": {"video": 1.0, "image": 0.8, "text": 1.2},
        "engagement_history": [
            {"post_id": "p01", "action": "liked",   "topic": "tech"},
            {"post_id": "p04", "action": "liked",   "topic": "tech"},
            {"post_id": "p06", "action": "liked",   "topic": "tech"},
            {"post_id": "p39", "action": "clicked", "topic": "gaming"},
            {"post_id": "p42", "action": "liked",   "topic": "gaming"},
            {"post_id": "p22", "action": "skipped", "topic": "fashion"},
            {"post_id": "p11", "action": "skipped", "topic": "cooking"},
        ],
    },
    "user_hard_3": {
        "id": "user_hard_3",
        "archetype": "critical_user",
        "stated_interests": ["music", "cooking"],
        "stated_dislikes": ["clickbait"],
        "hidden_weights": {
            "tech": 0.4, "cooking": 0.7, "fitness": 0.5,
            "fashion": 0.3, "travel": 0.5, "politics": 0.4,
            "gaming": 0.3, "music": 0.85
        },
        "content_type_preference": {"video": 1.0, "image": 1.1, "text": 1.0},
        "engagement_history": [
            {"post_id": "p45", "action": "liked",   "topic": "music"},
            {"post_id": "p48", "action": "liked",   "topic": "music"},
            {"post_id": "p09", "action": "liked",   "topic": "cooking"},
            {"post_id": "p17", "action": "skipped", "topic": "fitness"},
            {"post_id": "p05", "action": "skipped", "topic": "tech"},
            {"post_id": "p47", "action": "skipped", "topic": "music"},
        ],
    },
}


# ============================================================
# ENGAGEMENT MODEL — computes how likely a user engages with a post
# ============================================================

def compute_relevance(user: dict, post: dict) -> float:
    """
    Compute a relevance score (0.0 to 1.0) for how well a post
    matches a user's hidden preferences.

    This is the "ground truth" the agent is trying to predict.
    Higher = user would engage more with this post.
    """

    # Base relevance from topic preference
    topic_weight = user["hidden_weights"].get(post["topic"], 0.2)

    # Content type multiplier
    ct_mult = user.get("content_type_preference", {}).get(post["content_type"], 1.0)

    # Quality bonus — higher quality posts get engagement boost
    quality_bonus = post["quality_score"] * 0.3

    # Clickbait penalty — critical users penalize clickbait heavily
    clickbait_penalty = 0.0
    if post["is_clickbait"]:
        if user.get("archetype") == "critical_user":
            clickbait_penalty = -0.4
        else:
            clickbait_penalty = -0.1   # slight penalty even for casual users

    # Freshness bonus — newer posts get a small boost
    age = post.get("age_hours", 10)
    freshness = max(0.0, 1.0 - (age / 72.0)) * 0.15  # decays over 72 hours

    # New creator bonus — slight boost for discovery
    new_creator_bonus = 0.05 if post["is_new_creator"] else 0.0

    # Combine all factors
    score = (topic_weight * 0.5 * ct_mult) + quality_bonus + clickbait_penalty + freshness + new_creator_bonus

    # Clamp to 0-1 range
    return round(max(0.0, min(1.0, score)), 4)


def compute_ideal_ranking(user: dict, posts: list) -> list:
    """
    Return posts sorted by their true relevance to the user (best first).
    This is the 'ideal ranking' used to compute NDCG.
    """
    scored = [(post, compute_relevance(user, post)) for post in posts]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def compute_dcg(relevances: list, k: int = None) -> float:
    """
    Compute Discounted Cumulative Gain.
    DCG = sum of relevance[i] / log2(i + 2) for i in range(k)
    """
    if k is None:
        k = len(relevances)
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg


def compute_ndcg(user: dict, ranked_post_ids: list, all_posts: list, k: int = None) -> float:
    """
    Compute Normalized DCG for the agent's ranking vs ideal ranking.

    Args:
        user: user profile with hidden_weights
        ranked_post_ids: list of post IDs in the order the agent ranked them
        all_posts: the candidate post pool
        k: evaluate only top-k positions (None = all)

    Returns:
        NDCG score between 0.0 and 1.0
    """
    # Build post lookup
    post_map = {p["id"]: p for p in all_posts}

    # Get relevance scores for agent's ranking
    agent_relevances = []
    for pid in ranked_post_ids:
        if pid in post_map:
            agent_relevances.append(compute_relevance(user, post_map[pid]))
        else:
            agent_relevances.append(0.0)

    # Get ideal relevances (best possible ranking)
    ideal_scored = compute_ideal_ranking(user, all_posts)
    ideal_relevances = [score for _, score in ideal_scored]

    # Compute DCGs
    if k is None:
        k = len(ranked_post_ids)

    dcg = compute_dcg(agent_relevances, k)
    idcg = compute_dcg(ideal_relevances, k)

    if idcg == 0:
        return 0.0

    return round(dcg / idcg, 4)


# ============================================================
# POLICY CHECKER — for hard task constraints
# ============================================================

HARD_TASK_POLICIES = {
    "min_diversity_topics": 3,         # at least 3 different topics in top 10
    "max_same_author_in_top5": 2,      # no more than 2 posts from same author in top 5
    "min_new_creator_posts": 1,        # at least 1 new creator in top 10
    "max_clickbait_in_top5": 1,        # no more than 1 clickbait in top 5
}


def check_policies(ranked_post_ids: list, all_posts: list, k_top: int = 10, k_author: int = 5) -> dict:
    """
    Check if the ranking satisfies platform policies.

    Returns dict with:
        - each policy: True/False
        - policy_score: 0.0 to 1.0 (fraction of policies satisfied)
    """
    post_map = {p["id"]: p for p in all_posts}
    ranked_posts = [post_map[pid] for pid in ranked_post_ids if pid in post_map]

    top_k = ranked_posts[:k_top]
    top_5 = ranked_posts[:k_author]

    results = {}

    # Policy 1: topic diversity in top 10
    topics_in_top = set(p["topic"] for p in top_k)
    results["diversity"] = len(topics_in_top) >= HARD_TASK_POLICIES["min_diversity_topics"]

    # Policy 2: max same author in top 5
    author_counts = {}
    for p in top_5:
        author_counts[p["author_id"]] = author_counts.get(p["author_id"], 0) + 1
    max_author = max(author_counts.values()) if author_counts else 0
    results["author_spread"] = max_author <= HARD_TASK_POLICIES["max_same_author_in_top5"]

    # Policy 3: new creator representation in top 10
    new_creator_count = sum(1 for p in top_k if p["is_new_creator"])
    results["new_creators"] = new_creator_count >= HARD_TASK_POLICIES["min_new_creator_posts"]

    # Policy 4: clickbait limit in top 5
    clickbait_count = sum(1 for p in top_5 if p["is_clickbait"])
    results["clickbait_limit"] = clickbait_count <= HARD_TASK_POLICIES["max_clickbait_in_top5"]

    # Overall policy score
    passed = sum(1 for v in results.values() if v)
    results["policy_score"] = round(passed / len(results), 4)

    return results


# ============================================================
# TASK DEFINITIONS
# ============================================================

TASKS = {
    "easy": {
        "description": (
            "Rank 15 candidate posts for a single user with EXPLICIT preferences. "
            "The user's interests and dislikes are clearly stated. "
            "Select and rank the top 10 posts to maximize engagement."
        ),
        "action_schema": {
            "ranked_post_ids": "list of 10 post IDs in ranked order (best first)"
        },
        "num_candidates": 15,
        "top_k": 10,
        "user_ids": ["user_easy"],
        "policies_enforced": False,
    },

    "medium": {
        "description": (
            "Rank 25 candidate posts for a single user with NO stated preferences. "
            "You must infer their interests from their engagement history. "
            "Select and rank the top 10 posts to maximize engagement."
        ),
        "action_schema": {
            "ranked_post_ids": "list of 10 post IDs in ranked order (best first)"
        },
        "num_candidates": 25,
        "top_k": 10,
        "user_ids": ["user_medium"],
        "policies_enforced": False,
    },

    "hard": {
        "description": (
            "Rank 20 candidate posts for 3 different users sequentially. "
            "Some users have stated preferences, others only have history. "
            "You MUST also satisfy platform policies: "
            "minimum 3 topics in top 10, max 2 posts from same author in top 5, "
            "at least 1 new creator in top 10, max 1 clickbait in top 5. "
            "Score = 0.6 * average_NDCG + 0.4 * policy_compliance."
        ),
        "action_schema": {
            "ranked_post_ids": "list of 10 post IDs in ranked order (best first)"
        },
        "num_candidates": 20,
        "top_k": 10,
        "user_ids": ["user_hard_1", "user_hard_2", "user_hard_3"],
        "policies_enforced": True,
    },
}


# ============================================================
# HELPER — select candidate posts for a task
# ============================================================

def select_candidates(num: int, seed: int = None) -> list:
    """Select a random subset of posts as candidates for ranking."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    return rng.sample(POSTS, min(num, len(POSTS)))

# ============================================================
# ADD THIS TO THE BOTTOM OF YOUR data.py
# Place it BEFORE the if __name__ == "__main__": block
# ============================================================


# ============================================================
# DYNAMIC POST GENERATION — no two episodes are identical
# ============================================================



POST_TITLES = {
    "tech": [
        "New breakthrough in quantum computing",
        "Why Rust is replacing C++ in 2026",
        "Building AI agents from scratch",
        "Open source LLM comparison guide",
        "Smart home automation on a budget",
        "The future of wearable tech",
        "Database optimization deep dive",
        "Cybersecurity threats you should know",
        "Cloud vs edge computing explained",
        "The rise of decentralized apps",
    ],
    "cooking": [
        "One-pot pasta recipes for lazy nights",
        "Mastering sourdough from scratch",
        "10 spice blends every kitchen needs",
        "Farm to table — growing your own herbs",
        "Japanese curry from scratch",
        "Meal prep hacks that save hours",
        "The science behind perfect steak",
        "Vegan desserts that fool everyone",
        "Regional Indian cooking guide",
        "Homemade hot sauce recipes",
    ],
    "fitness": [
        "Bodyweight workout for beginners",
        "Running your first 5K — complete plan",
        "Mobility exercises for desk workers",
        "Progressive overload explained simply",
        "Recovery and rest day strategies",
        "Jump rope vs running — which burns more",
        "Building a calisthenics routine",
        "Stretching myths debunked",
        "Trail running gear essentials",
        "How sleep affects muscle growth",
    ],
    "fashion": [
        "Building a versatile work wardrobe",
        "Vintage shopping tips and tricks",
        "Color theory for outfit planning",
        "Ethical brands worth supporting",
        "Seasonal transition outfit ideas",
        "DIY clothing alterations guide",
        "Accessorizing on a budget",
        "Street style inspiration roundup",
        "Fabric care and longevity tips",
        "Gender neutral fashion trends",
    ],
    "travel": [
        "Off-season travel destinations 2026",
        "Packing light — the ultimate guide",
        "Digital nomad cities ranked",
        "Cultural etiquette tips by country",
        "Adventure travel on a budget",
        "Train journeys worth the detour",
        "Photography tips for travelers",
        "Sustainable tourism practices",
        "Solo travel safety essentials",
        "Hidden food markets worldwide",
    ],
    "politics": [
        "Understanding local ballot measures",
        "How policy affects housing prices",
        "Youth voter engagement strategies",
        "Data privacy legislation overview",
        "Climate policy comparison by country",
        "Healthcare system reform explained",
        "Education funding analysis",
        "Infrastructure spending breakdown",
        "International trade agreements simplified",
        "Civic participation guide for beginners",
    ],
    "gaming": [
        "Indie game gems of 2026",
        "Building a streaming setup guide",
        "Game design principles for beginners",
        "Retro gaming collection tips",
        "Esports career paths explained",
        "Modding community spotlight",
        "VR gaming state of the art",
        "Couch co-op games ranked",
        "Game soundtrack appreciation",
        "Accessibility in modern games",
    ],
    "music": [
        "Home recording studio setup guide",
        "Learning music theory from zero",
        "Playlist curation as an art form",
        "Live performance tips for beginners",
        "Sound design basics explained",
        "Building a vinyl collection",
        "Music production on a budget",
        "Genre deep dive — lo-fi origins",
        "Ear training exercises daily",
        "Collaborative music making online",
    ],
}

CLICKBAIT_TEMPLATES = [
    "You WON'T BELIEVE what happened with {topic}!!!",
    "This {topic} secret will CHANGE YOUR LIFE!!!",
    "{topic} experts HATE this one trick!!!",
    "SHOCKING {topic} discovery revealed!!!",
    "Stop everything — {topic} will NEVER be the same!!!",
]

AUTHOR_POOL = [f"author_{i:03d}" for i in range(1, 51)]


def generate_dynamic_posts(num_posts: int = 20, seed: int = None) -> list:
    """
    Generate a fresh set of posts with randomized features.
    No two episodes will have identical posts — agents can't memorize answers.
    """
    if seed is None:
        seed = int(time.time() * 1000) % 100000
    rng = random.Random(seed)

    posts = []
    topics = list(POST_TITLES.keys())

    for i in range(num_posts):
        topic = rng.choice(topics)
        is_clickbait = rng.random() < 0.2  # 20% chance of clickbait

        if is_clickbait:
            template = rng.choice(CLICKBAIT_TEMPLATES)
            title = template.format(topic=topic.capitalize())
            quality = round(rng.uniform(0.1, 0.4), 2)
        else:
            title = rng.choice(POST_TITLES[topic])
            quality = round(rng.uniform(0.5, 0.95), 2)

        post = {
            "id": f"d{i+1:02d}",
            "title": title,
            "topic": topic,
            "content_type": rng.choice(CONTENT_TYPES),
            "author_id": rng.choice(AUTHOR_POOL),
            "author_popularity": round(rng.uniform(0.2, 0.95), 2),
            "is_new_creator": rng.random() < 0.25,  # 25% new creators
            "quality_score": quality,
            "is_clickbait": is_clickbait,
            "age_hours": rng.randint(1, 72),
        }
        posts.append(post)

    return posts


# ============================================================
# RICHER USER SIMULATION — adds realistic noise
# ============================================================

def compute_relevance_with_noise(user: dict, post: dict, noise_seed: int = None) -> float:
    """
    Same as compute_relevance but with stochastic noise.
    Real users don't behave identically every time — this simulates
    mood, attention, and context variability.

    Noise is seeded for reproducibility during grading.
    """
    base_score = compute_relevance(user, post)

    # Add bounded Gaussian noise
    if noise_seed is not None:
        rng = random.Random(noise_seed)
    else:
        rng = random.Random()

    noise = rng.gauss(0, 0.05)  # small noise, std dev = 0.05
    noisy_score = base_score + noise

    # Clamp to 0-1
    return round(max(0.0, min(1.0, noisy_score)), 4)


def simulate_engagement(user: dict, ranked_posts: list, noise_seed: int = 42) -> list:
    """
    Simulate how a user would actually engage with a ranked feed.
    Returns a list of engagement events (click/like/skip/ignore).

    This provides richer feedback than just a score — agents can
    learn from the pattern of engagement.
    """
    engagements = []
    rng = random.Random(noise_seed)

    for position, post in enumerate(ranked_posts):
        relevance = compute_relevance_with_noise(
            user, post, noise_seed=noise_seed + position
        )

        # Position decay — users less likely to engage with lower-ranked posts
        position_factor = 1.0 / (1.0 + 0.1 * position)
        engage_prob = relevance * position_factor

        # Determine engagement type based on probability
        roll = rng.random()
        if roll < engage_prob * 0.4:
            action = "liked"
        elif roll < engage_prob * 0.8:
            action = "clicked"
        elif roll < engage_prob:
            action = "viewed"
        else:
            action = "skipped"

        engagements.append({
            "post_id": post["id"],
            "position": position + 1,
            "action": action,
            "relevance": relevance,
        })

    return engagements


# ============================================================
# ENHANCED CANDIDATE SELECTION — mix static + dynamic posts
# ============================================================

def select_candidates_enhanced(num: int, seed: int = None, dynamic_ratio: float = 0.4) -> list:
    """
    Select candidates mixing static posts (from POSTS) with
    dynamically generated posts. This ensures agents face
    both familiar and novel content each episode.

    dynamic_ratio: fraction of posts that are dynamically generated (0.0 to 1.0)
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    num_dynamic = int(num * dynamic_ratio)
    num_static = num - num_dynamic

    # Select static posts
    static = rng.sample(POSTS, min(num_static, len(POSTS)))

    # Generate dynamic posts
    dynamic = generate_dynamic_posts(num_dynamic, seed=seed + 999 if seed else None)

    # Combine and shuffle
    combined = static + dynamic
    rng.shuffle(combined)

    return combined[:num]

# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    print(f"Total posts: {len(POSTS)}")
    print(f"Topics: {TOPICS}")
    print(f"Users: {list(USERS.keys())}")
    print(f"Tasks: {list(TASKS.keys())}")
    print()

    # Test relevance scoring
    user = USERS["user_easy"]
    print(f"User: {user['id']} (likes: {user['stated_interests']})")
    print("Top 5 posts by relevance:")
    ideal = compute_ideal_ranking(user, POSTS)
    for post, score in ideal[:5]:
        print(f"  {post['id']}: {post['title'][:45]:45s} topic={post['topic']:10s} relevance={score}")

    print()
    print("Bottom 5 posts by relevance:")
    for post, score in ideal[-5:]:
        print(f"  {post['id']}: {post['title'][:45]:45s} topic={post['topic']:10s} relevance={score}")

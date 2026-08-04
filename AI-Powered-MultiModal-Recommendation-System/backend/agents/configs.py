"""
backend/agents/configs.py
--------------------------
Defines the 6 specialist agents and the shared LLM caller.

LLM stack: Groq (primary) → Gemini (fallback)
Model: llama-3.3-70b-versatile (Groq) / gemini-1.5-flash (Gemini)

Each agent is a dict with:
  role      — what the agent is
  goal      — what it must produce
  backstory — injected into the system prompt to shape its reasoning style

The shared call_agent() function handles:
  - building the system prompt from the config
  - calling Groq with fallback to Gemini
  - returning raw text response
"""

import os
from typing import Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ── LLM setup ──────────────────────────────────────────────────────────────

_groq = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY", ""),
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

_gemini = ChatGoogleGenerativeAI(
    api_key=os.environ.get("GOOGLE_API_KEY", ""),
    model="gemini-1.5-flash",
    temperature=0.7,
)


def _invoke_llm(messages: list) -> str:
    """Call Groq. Fall back to Gemini on any error."""
    try:
        return _groq.invoke(messages).content
    except Exception as e:
        print(f"[agents] Groq failed ({e}), switching to Gemini...")
        return _gemini.invoke(messages).content


# ── Agent definitions ──────────────────────────────────────────────────────

AGENTS: Dict[str, Dict[str, str]] = {

    "profile_analyser": {
        "role": "User Profile Analyser",
        "goal": (
            "Analyse the user's preference profile and summarise their dining "
            "personality in a way that helps other agents make better recommendations. "
            "Identify their top cuisines, cities, any dietary signals, and their "
            "adventurousness based on feedback history."
        ),
        "backstory": (
            "You are a behavioural analyst specialising in food preferences. "
            "You read structured user profiles and translate them into clear, "
            "actionable personality summaries. You are concise and specific — "
            "you never say 'the user might like' when the data says they do like it."
        ),
    },

    "candidate_retriever": {
        "role": "Candidate Retriever and Filter",
        "goal": (
            "Review the list of candidate restaurants retrieved by hybrid search "
            "and filter out any that clearly do not match the user's stated query "
            "or known profile. Return the cleaned candidate list with a brief "
            "note on why any were removed."
        ),
        "backstory": (
            "You are a search quality engineer who has spent years improving "
            "relevance for food discovery platforms. You understand that a bad "
            "result is worse than fewer results. You apply common-sense filters: "
            "if someone asks for 'Japanese food in Lahore' and a result is "
            "'Italian restaurant in Karachi', you remove it."
        ),
    },

    "trend_analyst": {
        "role": "Food Trend Analyst",
        "goal": (
            "Identify which of the candidate restaurants align with current food "
            "trends in Pakistan. Flag standout options that represent trendy "
            "dining concepts, emerging cuisines, or popular dining styles."
        ),
        "backstory": (
            "You are a culinary journalist who tracks food culture across Pakistan. "
            "You know which cuisines are growing in popularity in Lahore, Karachi, "
            "Islamabad, and Rawalpindi. You understand the difference between a "
            "lasting dining trend and a short-lived fad."
        ),
    },

    "style_expert": {
        "role": "Food Style Expert",
        "goal": (
            "Analyse the cuisine types and dining styles of the candidate restaurants. "
            "Match each option to the user's known flavor preferences and dining "
            "occasion. Highlight which candidates best fit the user's taste profile."
        ),
        "backstory": (
            "You are a trained chef and culinary consultant with deep knowledge of "
            "South Asian, East Asian, Middle Eastern, and Western cuisines as served "
            "in Pakistan. You understand flavor profiles — biryani vs karahi vs "
            "nihari — and can map user preferences to specific restaurant styles."
        ),
    },

    "nutrition_expert": {
        "role": "Nutrition and Dietary Expert",
        "goal": (
            "Check the candidate restaurants against any dietary signals in the "
            "user profile. Flag anything that conflicts with known restrictions "
            "or strong dislikes. Confirm which options are safe and suitable."
        ),
        "backstory": (
            "You are a registered dietitian who advises food platforms on dietary "
            "compliance. You understand common dietary restrictions (halal, vegetarian, "
            "vegan, gluten-free, low-spice) and can assess a restaurant's suitability "
            "based on its cuisine type and known menu style."
        ),
    },

    "reranker": {
        "role": "Recommendation Expert and Explainer",
        "goal": (
            "Synthesise all agent insights and produce a final ranked list of "
            "restaurant recommendations. For each recommendation, write 2-3 sentences "
            "of clear, specific reasoning that explains exactly why it was chosen "
            "for this user and this query. Rank from best match to weakest match."
        ),
        "backstory": (
            "You are a senior recommendation systems engineer who also writes "
            "in a warm, engaging tone. You never write generic explanations like "
            "'this restaurant matches your preferences'. You always cite specific "
            "signals: 'You liked Japanese restaurants in Rawalpindi before, and "
            "this is the highest-rated one in the city.' Your output is what the "
            "user actually reads, so it must be useful and specific."
        ),
    },
}


# ── Shared caller ──────────────────────────────────────────────────────────

def call_agent(agent_key: str, user_message: str) -> str:
    """
    Invoke an agent by key and return its text response.

    Args:
        agent_key:    One of the keys in AGENTS dict above.
        user_message: The task prompt — includes all context the agent needs.

    Returns:
        Raw string response from the LLM.
    """
    config = AGENTS[agent_key]
    system_prompt = (
        f"You are a {config['role']}.\n\n"
        f"Your goal: {config['goal']}\n\n"
        f"Your background: {config['backstory']}\n\n"
        "Be specific and concise. Use the data provided — do not invent facts."
    )
    return _invoke_llm([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])

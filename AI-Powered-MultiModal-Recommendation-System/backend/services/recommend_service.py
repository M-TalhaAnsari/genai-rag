"""
backend/services/recommend_service.py
---------------------------------------
Thin wrapper around the agent workflow.
Handles the async → sync bridge (workflow is sync due to ThreadPoolExecutor).
Also logs the result and saves to long-term memory.
"""

import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.workflow import run_recommendation_workflow
from backend.services import analytics_service, memory_service


async def run_recommendation(
    db: AsyncSession,
    query: str,
    user_id: str | None,
    profile: dict | None,
    top_k: int = 5,
) -> dict:
    """
    Run the full 6-agent workflow and return the result state.

    The workflow is CPU-bound (LLM calls + ThreadPoolExecutor).
    We run it in an executor so we don't block the event loop.

    Returns the full state dict — routers slice out what they need.
    """
    loop = asyncio.get_event_loop()

    result_state = await loop.run_in_executor(
        None,
        lambda: run_recommendation_workflow(
            query=query,
            user_id=user_id or "anonymous",
            profile=profile,
        ),
    )

    recommendations = result_state.get("final_recommendations", [])[:top_k]

    # Persist conversation turn
    if user_id:
        assistant_response = (
            f"Found {len(recommendations)} recommendations for: {query}"
        )
        await memory_service.save_conversation(
            db=db,
            user_id=user_id,
            user_message=query,
            assistant_response=assistant_response,
            query=query,
        )
        memory_service.add_message(user_id, "assistant", assistant_response)

    # Log to analytics
    await analytics_service.log_search(
        db=db,
        query=query,
        result_count=len(recommendations),
        user_id=user_id,
    )

    result_state["final_recommendations"] = recommendations
    return result_state
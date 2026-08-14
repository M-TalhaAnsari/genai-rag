"""
backend/routers/recommend.py
------------------------------
POST /recommend — full 6-agent pipeline with SSE streaming.
"""

import json
import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.models.schemas import RecommendRequest
from backend.retrieving import vector_store
from backend.services import feedback_service as feedback_module
from backend.services import analytics_service as analytics_module
from backend.agents.workflow import run_recommendation_workflow
from backend.memory import session as session_memory
from backend.memory.long_term import save_conversation

from fastapi import APIRouter, Depends
from backend.core.security import get_current_user
from backend.models.db_models import User


router = APIRouter(tags=["recommend"])


@router.post("/recommend")
async def recommend(
    request: RecommendRequest,
    db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """
    Full 6-agent recommendation pipeline with SSE streaming.

    Streams phase updates so the user sees progress in real time
    rather than waiting 30 seconds for a response.

    Event types:
      {"event": "phase",  "message": "Analysing your profile..."}
      {"event": "result", "recommendations": [...], ...}
      {"event": "done"}
    """
    if vector_store.collection_count() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No restaurants indexed. Run /ingestion/restaurant-sync first."
        )

    user_id = current_user.id
    user_id_Str = str(current_user.id)

    profile = await feedback_module.get_profile(db, user_id)
    session_memory.add_query(user_id_Str, request.query)
    session_memory.add_message(user_id_Str, "user", request.query)

    async def event_stream():
        phases = [
            "Analysing your profile...",
            "Retrieving candidates...",
            "Running trend, style and nutrition analysis...",
            "Generating personalised recommendations..."
        ]

        for phase_msg in phases:
            yield f"data: {json.dumps({'event': 'phase', 'message': phase_msg})}\n\n"
            await asyncio.sleep(0.05)

        loop = asyncio.get_event_loop()
        result_state = await loop.run_in_executor(
            None,
            lambda: run_recommendation_workflow(
                query=request.query,
                user_id=request.user_id or "anonymous",
                profile=profile
            )
        )

        recommendations = result_state.get("final_recommendations", [])[:request.top_k]

        assistant_response = (
                        f"Found {len(recommendations)} recommendations for: {request.query}"
                    )
        
        await save_conversation(
            db=db,
            user_id=user_id,
            user_message=request.query,
            assistant_response=assistant_response,
            query=request.query
        )
        session_memory.add_message(user_id_Str, "assistant", assistant_response)

        await analytics_module.log_search(
            db=db, query=request.query,
            result_count=len(recommendations),
            user_id=user_id
        )

        yield f"data: {json.dumps({'event': 'result', 'query': request.query, 'user_id': user_id_str, 'personalised': profile is not None, 'result_count': len(recommendations), 'recommendations': recommendations, 'debug': {'profile_summary': result_state.get('profile_summary', ''), 'candidates_after_filter': len(result_state.get('filtered_candidates', []))}})}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"
 
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
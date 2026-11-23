"""
LIFEOS Agentic AI REST API

Endpoints for the conversational agentic financial assistant:
  POST /api/agent/chat           — Send a message, get a grounded response
  GET  /api/agent/conversations  — List past conversations (stub for future persistence)
  GET  /api/agent/tools          — Introspect available tool schemas
"""

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db
from app.models.models import User
from app.schemas.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentToolSchema,
)
from app.services.auth_service import get_current_user
from app.services.agent_service import AgentService
from app.services.agent_tools import get_tool_schemas

router = APIRouter(prefix="/api/agent", tags=["Agentic AI"])


@router.post("/chat", response_model=AgentChatResponse, status_code=status.HTTP_200_OK)
async def agent_chat(
    request: AgentChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a natural-language message to the LIFEOS agentic assistant.

    The agent:
    1. Classifies intent from the user's message.
    2. Selects and invokes appropriate typed tools against real financial data.
    3. Returns a grounded, evidence-backed response — no hallucinated numbers.

    Safety: The agent is READ-ONLY and cannot modify any financial records.
    """
    agent = AgentService(db, current_user.id)
    result = await agent.chat(
        message=request.message,
        conversation_id=request.conversation_id,
    )
    return AgentChatResponse(**result)


@router.get("/tools", response_model=List[AgentToolSchema])
async def list_agent_tools(
    current_user: User = Depends(get_current_user),
):
    """
    List all available agent tools with their name, description,
    parameter schema, and category. Useful for frontend introspection
    and developer documentation.
    """
    schemas = get_tool_schemas()
    return [AgentToolSchema(**s) for s in schemas]


@router.get("/conversations", response_model=List[Dict[str, Any]])
async def list_conversations(
    current_user: User = Depends(get_current_user),
):
    """
    List recent conversations for the authenticated user.
    Currently returns an empty list — conversation persistence is
    planned for a future phase.
    """
    # Stub: conversation persistence deferred to Phase 16+
    return []

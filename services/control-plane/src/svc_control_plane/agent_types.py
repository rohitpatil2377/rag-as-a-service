"""Shared types for the agent orchestrator (Agentic RAG with MCP + AI agents).

A goal is turned into a :class:`Plan` (planner role), each :class:`PlanStep` is a call to a named
MCP tool; a critic role reviews the plan (:class:`Review`); the executor runs the steps against the
MCP tool registry, gating CRITICAL steps behind human approval (dual-control), and records a
:class:`StepObservation` per step. The whole run is an :class:`AgentRunResult`.

These are the contracts the planner / guard / critic modules are built against, so the pieces
compose without importing each other.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """One step of an agent plan: call MCP tool ``tool`` with ``args``. ``rationale`` is the
    planner's one-line reason (kept for the audit/decision trace, never executed)."""

    tool: str = Field(..., description="Name of the MCP tool to call (must be in the tool catalog).")
    args: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool call.")
    rationale: str = Field(default="", description="Why this step — explainability, not executed.")


class Plan(BaseModel):
    """An ordered plan of tool calls to accomplish ``goal``."""

    goal: str
    steps: list[PlanStep] = Field(default_factory=list)


class StepObservation(BaseModel):
    """The outcome of executing one plan step (ReAct 'observation')."""

    tool: str
    ok: bool
    result: Any = Field(default=None, description="Tool result on success (JSON-serialisable).")
    error: str = Field(default="", description="Error message on failure.")
    approval_id: str | None = Field(default=None, description="Set when the step is awaiting human approval.")


class Review(BaseModel):
    """A critic role's verdict on a plan before it runs — the multi-agent reasoning check."""

    approved: bool = Field(..., description="False blocks execution until the plan is revised.")
    concerns: list[str] = Field(default_factory=list, description="Specific risks/gaps found.")
    suggestion: str = Field(default="", description="How to improve the plan, if any.")


class AgentRunResult(BaseModel):
    """The full record of an agent run: goal, plan, critic review, per-step observations, the
    synthesised answer, and — if a critical step is gated — the pending approval id."""

    goal: str
    plan: Plan
    review: Review | None = None
    observations: list[StepObservation] = Field(default_factory=list)
    answer: str = ""
    pending_approval_id: str | None = Field(
        default=None, description="Set when the run paused for human approval of a critical action."
    )
    halted_reason: str = Field(default="", description="Why the run stopped early (critic block / approval / error).")

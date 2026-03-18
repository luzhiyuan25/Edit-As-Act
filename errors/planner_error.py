"""Planner-specific error classes for EditAsAct."""


class PlannerSchemaOrLogicError(Exception):
    """Raised when the planner encounters a schema or logic error.
    
    This includes:
    - LLM returning malformed JSON
    - Invalid action schemas
    - Logic errors in planning (cycles, no progress, etc.)
    - Validation failures
    """
    pass


class PlannerMaxRetriesError(PlannerSchemaOrLogicError):
    """Raised when max retries exceeded during planning."""
    pass


class PlannerCycleError(PlannerSchemaOrLogicError):
    """Raised when a cycle is detected in the planning loop."""
    pass

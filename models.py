# models.py
"""
Data models for the Enterprise QA Environment.

Supports multi-domain strategic analysis tasks with rich observation fields.
"""

from typing import Any, Dict, Optional
from pydantic import Field
from openenv.core.envserver.types import Action, Observation


class EnterpriseQaAction(Action):
    """Agent's response to a task prompt."""
    message: str = Field(..., description="The agent's answer to the task")


class EnterpriseQaObservation(Observation):
    """
    Full observation returned after reset() or step().

    Fields:
        task_id:        Unique task identifier from data.csv
        domain:         Business domain (Finance, Healthcare, etc.)
        prompt:         The task description / question
        rubric:         Scoring criteria string
        echoed_message: The agent's last response (for debugging)
        done:           True when episode terminates
        reward:         Score in [0.0, 1.0]
        metadata:       Step count, gold_response, episode_id
    """
    task_id: str = Field(default="", description="Task identifier")
    domain: str = Field(default="", description="Business domain of the task")
    prompt: str = Field(default="", description="Task prompt for the agent")
    rubric: str = Field(default="", description="Scoring rubric / criteria")
    echoed_message: str = Field(default="", description="Agent's last response")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Step metadata")

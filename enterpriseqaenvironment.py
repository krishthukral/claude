# server/enterpriseqaenvironment.py
"""
Enterprise QA Environment Implementation.

Serves multi-domain strategic analysis tasks from a CSV file.
Rewards agents based on numeric accuracy + rubric criteria matching.
"""

import csv
import logging
import os
import random
from uuid import uuid4
from typing import Any, Dict, List, Optional

from openenv.core.envserver.interfaces import Environment
from openenv.core.envserver.types import State

from models import EnterpriseQaAction, EnterpriseQaObservation
from server.rewards import calculate_reward

logger = logging.getLogger(__name__)

# Resolve data path relative to this file, overridable via env var
_DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")
DATA_PATH = os.environ.get("ENTERPRISE_DATA_PATH", _DEFAULT_DATA)
MAX_STEPS = int(os.environ.get("ENTERPRISE_MAX_STEPS", "5"))


class EnterpriseQaEnvironment(Environment):
    """
    Multi-domain enterprise QA environment.

    Each episode:
      1. reset() → returns a task prompt (market sizing, valuation, etc.)
      2. step()  → evaluates agent response, returns reward + done=True
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self.state: State = State(episode_id=str(uuid4()), step_count=0)
        self.tasks: List[Dict[str, Any]] = []
        self.current_task: Optional[Dict[str, Any]] = None
        self._shuffled: List[Dict[str, Any]] = []
        self._task_index: int = 0
        self._load_tasks()

    # ── Data Loading ──────────────────────────────────────────────────────────

    def _load_tasks(self) -> None:
        """Load and validate tasks from CSV."""
        try:
            with open(DATA_PATH, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.tasks = [row for row in reader if row.get("prompt", "").strip()]
            logger.info(f"Loaded {len(self.tasks)} tasks from {DATA_PATH}")
        except FileNotFoundError:
            logger.error(f"Data file not found: {DATA_PATH}")
            self.tasks = [{
                "task_id": "fallback",
                "domain": "General",
                "prompt": "No tasks loaded. Check ENTERPRISE_DATA_PATH.",
                "gold_response": "",
                "rubric": "",
            }]
        self._shuffle_tasks()

    def _shuffle_tasks(self) -> None:
        self._shuffled = self.tasks.copy()
        random.shuffle(self._shuffled)
        self._task_index = 0

    def _next_task(self) -> Dict[str, Any]:
        if self._task_index >= len(self._shuffled):
            self._shuffle_tasks()
        task = self._shuffled[self._task_index]
        self._task_index += 1
        return task

    # ── Environment API ───────────────────────────────────────────────────────

    def reset(self, domain: Optional[str] = None, **kwargs) -> EnterpriseQaObservation:
        """
        Reset environment and serve next task.

        Args:
            domain: Optional domain filter (e.g. "Finance", "Healthcare")
        """
        self.state = State(episode_id=str(uuid4()), step_count=0)

        if domain:
            pool = [t for t in self.tasks if t.get("domain", "").lower() == domain.lower()]
            self.current_task = random.choice(pool) if pool else self._next_task()
        else:
            self.current_task = self._next_task()

        logger.info(
            f"Episode {self.state.episode_id} | "
            f"Task {self.current_task.get('task_id', '?')} | "
            f"Domain: {self.current_task.get('domain', '?')}"
        )

        return EnterpriseQaObservation(
            task_id=self.current_task.get("task_id", ""),
            domain=self.current_task.get("domain", ""),
            prompt=self.current_task.get("prompt", ""),
            rubric=self.current_task.get("rubric", ""),
            echoed_message="",
            done=False,
            reward=0.0,
        )

    def step(self, action: EnterpriseQaAction, **kwargs) -> EnterpriseQaObservation:  # type: ignore[override]
        """Evaluate agent response and return reward."""
        self.state.step_count += 1

        if not self.current_task:
            return EnterpriseQaObservation(
                task_id="",
                domain="",
                prompt="No active task. Call reset() first.",
                rubric="",
                echoed_message=action.message,
                done=True,
                reward=0.0,
            )

        gold = self.current_task.get("gold_response", "")
        rubric = self.current_task.get("rubric", "")
        reward = calculate_reward(action.message, gold, rubric=rubric)

        # Episode ends after answer submission OR max steps
        done = True if self.state.step_count >= MAX_STEPS else True  # single-step tasks always done

        logger.info(
            f"Episode {self.state.episode_id} | Step {self.state.step_count} | "
            f"Reward: {reward:.4f} | Done: {done}"
        )

        return EnterpriseQaObservation(
            task_id=self.current_task.get("task_id", ""),
            domain=self.current_task.get("domain", ""),
            prompt=self.current_task.get("prompt", ""),
            rubric=rubric,
            echoed_message=action.message,
            done=done,
            reward=reward,
            metadata={
                "step": self.state.step_count,
                "gold_response": gold,
                "episode_id": self.state.episode_id,
            },
        )

    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, value: State) -> None:
        self._state = value
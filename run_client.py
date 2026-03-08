# run_client.py
"""
Test client for the Enterprise QA environment.

Runs through all tasks in data.csv, submits gold responses,
and reports per-task and aggregate rewards.
"""

import pandas as pd
from enterpriseqa import EnterpriseQaAction, EnterpriseQaEnv

PORT = 8001
DATA_FILE = "data/data.csv"


def run_all_tasks():
    try:
        df = pd.read_csv(DATA_FILE)
        df["prompt"] = df["prompt"].str.strip()
    except FileNotFoundError:
        print(f"ERROR: {DATA_FILE} not found.")
        return

    results = []

    with EnterpriseQaEnv(base_url=f"http://localhost:{PORT}") as env:
        for _, row in df.iterrows():
            # Reset and verify we got the right task
            obs = env.reset()
            prompt = obs.observation.prompt.strip()
            task_id = obs.observation.task_id
            domain = obs.observation.domain

            # Match task by task_id (robust) or prompt fallback
            match = df[df["task_id"] == task_id]
            if match.empty:
                match = df[df["prompt"].str.strip() == prompt]

            if match.empty:
                print(f"[WARN] Could not match task_id={task_id}")
                continue

            gold = match.iloc[0]["gold_response"]
            action = EnterpriseQaAction(message=gold)
            result = env.step(action)

            reward = result.reward
            results.append({
                "task_id": task_id,
                "domain": domain,
                "reward": reward,
                "passed": reward >= 0.7,
            })

            status = "✅" if reward >= 0.7 else "❌"
            print(f"{status} [{domain}] {task_id[:20]}... → Reward: {reward:.4f}")

    # Summary
    if results:
        avg = sum(r["reward"] for r in results) / len(results)
        passed = sum(1 for r in results if r["passed"])
        print(f"\n{'='*50}")
        print(f"Tasks: {len(results)} | Passed: {passed} | Avg Reward: {avg:.4f}")
        print(f"Pass Rate: {passed/len(results)*10


import subprocess
import os

tasks = ["agentic", "analysis", "classification", "code", "extraction", "math", "qa"]
model = "gpt-4o"

env = os.environ.copy()
env["DSP_CACHEBOOL"] = "False"

for task in tasks:
    print(f"Restoring {model} on {task}...")
    cmd = [
        "python3", "src/run_baseline_v2.py",
        "--filter-model", model,
        "--filter-task", task
    ]
    subprocess.run(cmd, env=env)

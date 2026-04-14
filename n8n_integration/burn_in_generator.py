"""
Burn-in Generator for n8n Real Traces
Generates the 50 burn-in events to POST to /init before a real Claude experiment.
These establish the A₀ snapshot using realistic baseline tool distribution.

Usage:
    python burn_in_generator.py | python -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))"

Or: POST directly to IML node
    python burn_in_generator.py --post http://localhost:5050/init --agent_id agent_A
"""
import json
import random
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from iml.deviation import ALL_TOOLS

# Admission-time distribution (matches benchmark base_probs)
BASE_PROBS = {
    "safe_read":      0.35,
    "safe_query":     0.35,
    "moderate_write": 0.12,
    "moderate_send":  0.08,
    "risky_execute":  0.05,
    "risky_delegate": 0.05,
}


def generate_burn_in(n: int = 50, seed: int = 42, agent_id: str = "agent_A"):
    random.seed(seed)
    tools = list(BASE_PROBS.keys())
    weights = list(BASE_PROBS.values())

    events = []
    for i in range(n):
        tool = random.choices(tools, weights=weights, k=1)[0]
        events.append({
            "tool":      tool,
            "depth":     1,
            "action":    "tool_call",
            "agent":     agent_id,
            "metadata":  {"source": "burn_in", "step": i}
        })

    return {
        "agent_id": agent_id,
        "burn_in":  events
    }


def post_to_server(payload: dict, url: str):
    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",        type=int,   default=50)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--agent_id", type=str,   default="agent_A")
    parser.add_argument("--post",     type=str,   default=None,
                        help="POST to this URL instead of printing")
    args = parser.parse_args()

    payload = generate_burn_in(n=args.n, seed=args.seed, agent_id=args.agent_id)

    if args.post:
        result = post_to_server(payload, args.post)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(payload, indent=2))

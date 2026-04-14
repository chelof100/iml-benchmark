"""
IML n8n Integration — Real Agent Trace Collection
Receives trace events from n8n webhooks, computes D̂ in real time,
and returns the IML signal for use in downstream n8n nodes.

Usage in n8n:
  - HTTP Request node POSTs trace events to this server
  - Code node calls the /compute endpoint
  - Results flow back into the n8n workflow

Start: python iml_node.py --port 5050
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request, jsonify
from iml.trace import Trace, Event
from iml.deviation import IML, AdmissionSnapshot
from baselines.enforcement import enforcement_signal
from baselines.anomaly import AnomalyDetector

app = Flask(__name__)

# ─── Global state (one session per agent_id) ─────────────────────────────────
sessions: dict = {}   # agent_id → {"iml": IML, "trace": Trace, "anomaly": AnomalyDetector}

def get_or_create_session(agent_id: str, burn_in_events=None):
    if agent_id not in sessions:
        if burn_in_events is None:
            raise ValueError(f"No session for '{agent_id}' — send burn_in first")
        burn_in_trace = Trace(trace_id=f"burn_in_{agent_id}")
        for ev in burn_in_events:
            burn_in_trace.add(Event(
                agent=ev.get("agent", "A"),
                action=ev.get("action", "tool_call"),
                tool=ev.get("tool"),
                depth=ev.get("depth", 1),
                metadata=ev.get("metadata", {})
            ))
        A0 = AdmissionSnapshot(burn_in_trace)
        sessions[agent_id] = {
            "A0":      A0,
            "iml":     IML(A0),
            "trace":   Trace(trace_id=agent_id),
            "anomaly": AnomalyDetector(window_size=30),
            "step":    0,
            "created": time.time(),
        }
        print(f"[IML] Session created: {agent_id}  |  burn_in={len(burn_in_events)} events")
    return sessions[agent_id]


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "sessions": list(sessions.keys())})


@app.route("/init", methods=["POST"])
def init_session():
    """
    Initialize a session with burn-in events.

    Body: {
      "agent_id": "agent_A",
      "burn_in": [
        {"tool": "safe_read", "depth": 1, "action": "tool_call"},
        ...
      ]
    }
    """
    data = request.json
    agent_id    = data.get("agent_id", "default")
    burn_in     = data.get("burn_in", [])

    if not burn_in:
        return jsonify({"error": "burn_in must be non-empty"}), 400

    # Remove existing session if re-init
    if agent_id in sessions:
        del sessions[agent_id]

    sess = get_or_create_session(agent_id, burn_in)
    A0 = sess["A0"]
    return jsonify({
        "status": "initialized",
        "agent_id": agent_id,
        "burn_in_steps": len(burn_in),
        "A0_depth_mean": round(A0.depth_mean, 4),
        "A0_depth_std":  round(A0.depth_std, 4),
        "A0_P0": {k: round(float(v), 4)
                  for k, v in zip(
                      ["safe_read","safe_query","moderate_write",
                       "moderate_send","risky_execute","risky_delegate"],
                      A0.P0.tolist())}
    })


@app.route("/event", methods=["POST"])
def process_event():
    """
    Process a single trace event and return D̂.

    Body: {
      "agent_id": "agent_A",
      "tool": "moderate_write",
      "depth": 2,
      "action": "tool_call",
      "metadata": {}
    }

    n8n mapping:
      {{ $json.tool }} → tool name from Claude tool_use block
      {{ $json.depth }} → delegation depth (count parent agents)
    """
    data     = request.json
    agent_id = data.get("agent_id", "default")
    tool     = data.get("tool", "safe_read")
    depth    = int(data.get("depth", 1))
    action   = data.get("action", "tool_call")
    metadata = data.get("metadata", {})

    try:
        sess = get_or_create_session(agent_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Add event to trace
    event = Event(agent=agent_id, action=action, tool=tool,
                  depth=depth, metadata=metadata)
    sess["trace"].add(event)
    sess["step"] += 1

    # Compute signals
    D_hat        = sess["iml"].compute(sess["trace"])
    D_t          = sess["iml"].D_t(sess["trace"])
    D_c          = sess["iml"].D_c(sess["trace"])
    D_l          = sess["iml"].D_l(sess["trace"])
    enforcement  = enforcement_signal(sess["trace"])
    anomaly_score = sess["anomaly"].update(sess["trace"])

    result = {
        "agent_id":    agent_id,
        "step":        sess["step"],
        "tool":        tool,
        "depth":       depth,
        "D_hat":       round(D_hat, 4),
        "D_t":         round(D_t, 4),
        "D_c":         round(D_c, 4),
        "D_l":         round(D_l, 4),
        "enforcement": enforcement,
        "anomaly":     round(anomaly_score, 4),
        "alert":       D_hat > 0.30,   # configurable threshold
    }

    # Log to stdout for n8n debugging
    if sess["step"] % 20 == 0 or result["alert"]:
        print(f"[{agent_id}] step={sess['step']:3d}  D_hat={D_hat:.4f}  "
              f"enforcement={enforcement}  alert={result['alert']}")

    return jsonify(result)


@app.route("/batch", methods=["POST"])
def process_batch():
    """
    Process multiple events in one call (useful for replay).

    Body: {
      "agent_id": "agent_A",
      "events": [{"tool": ..., "depth": ...}, ...]
    }
    """
    data     = request.json
    agent_id = data.get("agent_id", "default")
    events   = data.get("events", [])

    results = []
    for ev in events:
        resp = app.test_client().post("/event",
            data=json.dumps({"agent_id": agent_id, **ev}),
            content_type="application/json")
        results.append(resp.get_json())
    return jsonify(results)


@app.route("/state/<agent_id>", methods=["GET"])
def get_state(agent_id):
    """Return current session state for an agent."""
    if agent_id not in sessions:
        return jsonify({"error": "unknown agent_id"}), 404
    sess = sessions[agent_id]
    return jsonify({
        "agent_id":  agent_id,
        "step":      sess["step"],
        "created":   sess["created"],
        "trace_len": len(sess["trace"].events),
    })


@app.route("/reset/<agent_id>", methods=["DELETE"])
def reset_session(agent_id):
    """Reset a session (called when agent is re-admitted)."""
    if agent_id in sessions:
        del sessions[agent_id]
        return jsonify({"status": "reset", "agent_id": agent_id})
    return jsonify({"error": "unknown agent_id"}), 404


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    print(f"IML Node listening on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)

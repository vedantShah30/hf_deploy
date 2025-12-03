import runpod
import threading
import requests
from api_server import app

# -----------------------------
# Start Flask Server in Background
# -----------------------------
def start_flask():
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False,
        use_reloader=False
    )

threading.Thread(target=start_flask, daemon=True).start()


# -----------------------------
# RunPod Serverless Handler
# -----------------------------
def handler(event):
    """
    RunPod will call this with:
    {
        "endpoint": "/classify" | "/vqa" | "/grounding",
        "body": { ... }
    }
    """
    endpoint = event.get("endpoint")
    body = event.get("body", {})

    if not endpoint:
        return {"error": "Missing 'endpoint' in input"}

    try:
        url = f"http://localhost:7860{endpoint}"
        response = requests.post(url, json=body, timeout=300)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Start RunPod Serverless Engine
# -----------------------------
runpod.serverless.start({"handler": handler})

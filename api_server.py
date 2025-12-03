from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import InferencePipeline
import os
import requests
import uuid

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pipeline = None


def get_pipeline():
    """Lazily create a single global pipeline instance."""
    global pipeline
    if pipeline is None:
        pipeline = InferencePipeline()
    return pipeline


def _download_if_url(image_url: str) -> str:
    """
    If image_url is an http(s) URL, download it to UPLOAD_FOLDER and
    return the local file path. Otherwise assume it is already a path.
    """
    if image_url.startswith("http://") or image_url.startswith("https://"):
        filename = f"{uuid.uuid4().hex}.jpg"
        local_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return local_path
    return image_url


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ==========================================
# CORE ROUTES FOR FRONTEND
# ==========================================

@app.route('/classify', methods=['POST'])
def classify():
    """
    Input:  { "prompt": "..." }
    Output: { "category": "captioning" | "grounding" | "vqa", "raw_category": "...", "tasks": [...] }
    """
    try:
        data = request.get_json(silent=True) or {}
        prompt = (data.get('prompt') or '').strip()

        if not prompt:
            return jsonify({'status': 'error', 'message': 'Prompt required'}), 400

        result = get_pipeline().classify(prompt)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/vqa', methods=['POST'])
def vqa():
    """
    Input:  { "imageUrl": "...", "prompt": "..." }
    Output: { "answer": "...", "debug": { ... } }
    """
    try:
        data = request.get_json(silent=True) or {}
        image_url = (data.get('imageUrl') or '').strip()
        prompt = (data.get('prompt') or '').strip()

        if not image_url:
            return jsonify({'status': 'error', 'message': 'imageUrl required'}), 400
        if not prompt:
            return jsonify({'status': 'error', 'message': 'Prompt required'}), 400

        local_path = _download_if_url(image_url)
        if not os.path.exists(local_path):
            return jsonify({'status': 'error', 'message': 'Image not found'}), 404

        result = get_pipeline().vqa(local_path, prompt)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/grounding', methods=['POST'])
def grounding():
    """
    Input:  { "imageUrl": "...", "prompt": "..." }
    Output: { "coordinates": [...], "raw": ... }
    """
    try:
        data = request.get_json(silent=True) or {}
        image_url = (data.get('imageUrl') or '').strip()
        prompt = (data.get('prompt') or '').strip()

        if not image_url:
            return jsonify({'status': 'error', 'message': 'imageUrl required'}), 400
        if not prompt:
            return jsonify({'status': 'error', 'message': 'Prompt required'}), 400

        local_path = _download_if_url(image_url)
        if not os.path.exists(local_path):
            return jsonify({'status': 'error', 'message': 'Image not found'}), 404

        result = get_pipeline().grounding(local_path, prompt)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    # Bind to all interfaces so it works inside Docker / on VastAI
    app.run(host='0.0.0.0', port=7860, debug=False)

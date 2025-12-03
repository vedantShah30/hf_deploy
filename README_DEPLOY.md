## Visual API Server (`/classify`, `/vqa`, `/grounding`) – Full Guide

This server is a **Flask** app exposed via three main routes:

- **`POST /classify`** – decide if a prompt is `captioning`, `grounding`, or `vqa`
- **`POST /vqa`** – run VQA on an image + prompt
- **`POST /grounding`** – run spatial grounding and return coordinates

Everything below assumes you are inside the `testing` directory.

---

### 1) Run Locally (Windows PowerShell)

**1.1. Navigate to the project directory**

```powershell
cd C:\Users\Vedant\Desktop\ml\tati\testing
```

**1.2. Create a virtualenv and install dependencies**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements_api.txt
```

**1.3. Start the Flask server**

```powershell
python api_server.py
```

The server will start at `http://localhost:8000`.

Quick health check:

```powershell
curl http://localhost:8000/health
```

You should get:

```json
{"status": "ok"}
```

---

### 2) API Endpoints – Request/Response Format

All examples assume base URL `http://localhost:8000` (locally) or `http://<VAST_IP>:8000` on VastAI.

#### 2.1. `POST /classify`

- **Input JSON:**

```json
{
  "prompt": "Describe the scene in the image in detail."
}
```

- **Output JSON:**

```json
{
  "category": "captioning",
  "raw_category": "CAPTIONING",
  "tasks": []
}
```

- **Use on frontend:**
  - Send user query to `/classify`
  - If `category` is:
    - `"captioning"` → treat like VQA caption query (you can still call `/vqa`)
    - `"grounding"` → call `/grounding`
    - `"vqa"` → call `/vqa`

#### 2.2. `POST /vqa`

- **Input JSON:**

```json
{
  "imageUrl": "https://example.com/path/to/image.jpg",
  "prompt": "What is the main object in this image?"
}
```

- **Output JSON (simplified):**

```json
{
  "answer": "A large bridge over water",
  "image_type": "optical"
}
```

Notes:
- `imageUrl` can be:
  - An **HTTP/HTTPS URL** – the server downloads the image internally.
  - A **local path inside the container** – advanced usage.

#### 2.3. `POST /grounding`

- **Input JSON:**

```json
{
  "imageUrl": "https://example.com/path/to/image.jpg",
  "prompt": "Find the tennis court and return its location."
}
```

- **Output JSON (shape depends on your `run_grounding`):**

```json
{
  "coordinates": [
    [100, 150, 300, 400]
  ],
  "raw": [
    [100, 150, 300, 400]
  ]
}
```

You can treat `coordinates` as:

```text
[x_min, y_min, x_max, y_max]
```

and draw boxes on your frontend.

---

### 3) Build Docker Image Locally

Navigate **one level above** `testing` (i.e. project root) so the Docker build context is correct:

```bash
cd C:\Users\Vedant\Desktop\ml\tati
```

Build the Docker image (from WSL or a bash shell):

```bash
docker build -t tati-visual-api:latest -f testing/Dockerfile .
```

Run the container locally:

```bash
docker run --rm -p 8000:8000 tati-visual-api:latest
```

Then test from your host:

```bash
curl http://localhost:8000/health
```

---

### 4) Push Image and Deploy on VastAI

#### 4.1. Tag and push to a registry

Example for Docker Hub:

```bash
docker tag tati-visual-api:latest <your-docker-user>/tati-visual-api:latest
docker push <your-docker-user>/tati-visual-api:latest
```

Make sure `<your-docker-user>` is your Docker Hub username, and that the repo exists (or will be created on push).

#### 4.2. Launch on VastAI

1. Go to the VastAI UI.
2. Choose a **GPU instance** (CUDA 12.1 compatible).
3. In the **container image** field, set:
   - `<your-docker-user>/tati-visual-api:latest`
4. In **ports**:
   - Map container port `8000` to any external port (e.g. `8000:8000`).
5. Start the instance.

Once it’s running, VastAI will show you an external IP (and possibly a mapped port). The base URL will be:

```text
http://<VAST_INSTANCE_IP>:8000
```

You can verify:

```bash
curl http://<VAST_INSTANCE_IP>:8000/health
```

---

### 5) Example Frontend Calls

Assume you are using `fetch` in a web frontend and your VastAI instance URL is `http://<VAST_INSTANCE_IP>:8000`.

**Classify:**

```javascript
const res = await fetch("http://<VAST_INSTANCE_IP>:8000/classify", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ prompt: userPrompt })
});
const data = await res.json(); // data.category is "captioning" | "grounding" | "vqa"
```

**VQA:**

```javascript
const res = await fetch("http://<VAST_INSTANCE_IP>:8000/vqa", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ imageUrl, prompt: userPrompt })
});
const data = await res.json(); // data.answer
```

**Grounding:**

```javascript
const res = await fetch("http://<VAST_INSTANCE_IP>:8000/grounding", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ imageUrl, prompt: userPrompt })
});
const data = await res.json(); // data.coordinates
```

On any failure, the backend returns:

```json
{
  "status": "error",
  "message": "some error message..."
}
```

so you can display `message` directly in the UI.

---

### 6) Notes / Gotchas

- **Model load time**: first request may take a while as models are loaded.
- **GPU requirements**: this image expects a GPU with CUDA 12.1 + proper drivers (VastAI takes care of this).
- **Image URLs**: must be reachable from inside the VastAI container (public URLs).
- **No silent failures**: if something goes wrong, you will get a JSON with `status: "error"` and a text `message`.



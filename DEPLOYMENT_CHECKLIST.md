# geoNLI FastAPI Deployment Checklist

## 📁 Directory Structure

```
tati/
├── testing/                    ← All hosting files here
│   ├── api_server.py          ← FastAPI app
│   ├── inference.py           ← Your existing inference pipeline
│   ├── requirements_api.txt   ← API dependencies (fastapi, uvicorn, requests)
│   ├── Dockerfile            ← Container config
│   ├── README_DEPLOY.md      ← Full deployment guide
│   ├── DEPLOYMENT_CHECKLIST.md ← This file
│   └── [other .py files]
└── [other directories]
```

---

## 🚀 Local Testing (Windows)

### Step 1: Navigate to testing directory

```powershell
cd c:\Users\Vedant\Desktop\ml\tati\testing
```

### Step 2: Create virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
pip install -r requirements_api.txt
```

### Step 4: Start server

```powershell
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Step 5: Test endpoint (in a new PowerShell)

```powershell
# Health check
curl http://localhost:8000/health

# Test POST (minimal example)
$body = @{
    input_image = @{
        image_id = "test.png"
        image_url = "https://example.com/image.jpg"
    }
    queries = @{
        caption_query = @{
            instruction = "Describe this image"
        }
    }
} | ConvertTo-Json

curl -X POST http://localhost:8000/geoNLI/eval `
  -H "Content-Type: application/json" `
  -Body $body
```

---

## 🐳 Docker Setup (for VAST AI)

### Step 1: Build from tati root directory

```bash
cd c:\Users\Vedant\Desktop\ml\tati
docker build -t tati-geo-nli:latest -f testing/Dockerfile .
```

### Step 2: Test locally

```bash
docker run --rm -p 8000:8000 tati-geo-nli:latest
```

### Step 3: Push to Docker Hub

```bash
# First, create account on hub.docker.com and login
docker login

# Tag image
docker tag tati-geo-nli:latest <your-username>/tati-geo-nli:latest

# Push
docker push <your-username>/tati-geo-nli:latest
```

---

## ☁️ VAST AI Deployment

### Step 1: Log in to VAST AI

- Go to https://www.vastai.com
- Create account or sign in
- Go to "My Instances" → "Create Instance"

### Step 2: Configure Instance

- **GPU**: Select based on your inference speed needs (RTX 4090 for fastest, cheaper options available)
- **Docker Image**: `<your-username>/tati-geo-nli:latest` (or push to GitHub Container Registry if private)
- **Ports**: Map container port `8000` to host port (VAST will assign)
- **Disk Space**: Ensure enough for model weights (typically 10-50GB depending on models)

### Step 3: Launch & Connect

- VAST will show you the instance URL and port
- Access endpoint: `http://<VAST_INSTANCE_IP>:<PORT>/geoNLI/eval`

### Step 4: Send requests

```bash
curl -X POST http://<VAST_INSTANCE_IP>:<PORT>/geoNLI/eval \
  -H "Content-Type: application/json" \
  -d '{
    "input_image": {
      "image_url": "https://example.com/image.jpg"
    },
    "queries": {
      "caption_query": {
        "instruction": "Describe the scene"
      }
    }
  }'
```

---

## 🔧 Important Configuration Notes

### 1. Model Dependencies

The Dockerfile currently only installs FastAPI dependencies. You need to add your heavy dependencies. Edit `testing/Dockerfile` and add:

```dockerfile
# After "RUN pip install --no-cache-dir -r requirements_api.txt", add:

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other model dependencies
RUN pip install unsloth chromadb pillow

# Uncomment to install from Grounding/requirements.txt if needed
# RUN pip install --no-cache-dir -r ../Grounding/requirements.txt
```

### 2. Image Download Location

- Images from URLs are downloaded to: `./tmp_images/`
- Inside Docker: `/app/tmp_images/`
- These are temporary and can be cleaned up

### 3. Model Loading Timeout

- First request will be slow (model loading happens at startup)
- Subsequent requests will be faster
- On VAST AI, consider using persistent storage for model weights

### 4. Port Configuration

- Local: `http://localhost:8000`
- Docker: `0.0.0.0:8000` (mapped to host port)
- VAST AI: `http://<instance-ip>:<assigned-port>`

---

## 📊 Endpoint Reference

### Health Check

```
GET /health
Response: {"status": "ok", "pipeline_loaded": true/false}
```

### Main Inference Endpoint

```
POST /geoNLI/eval
Content-Type: application/json

Body:
{
  "input_image": {
    "image_id": "string",
    "image_url": "string (full URL)"
  },
  "queries": {
    "caption_query": { "instruction": "string" },
    "grounding_query": { "instruction": "string" },
    "attribute_query": {
      "binary": { "instruction": "string" },
      "numeric": { "instruction": "string" },
      "semantic": { "instruction": "string" }
    }
  }
}

Response (200 OK):
{
  "input": { ...echoed input... },
  "output": {
    "final_response": "string from InferencePipeline",
    "note": "description"
  }
}

Error (400): Missing or invalid input
Error (503): Models not loaded yet
Error (500): Inference failed
```

---

## 🐛 Troubleshooting

### Local Testing Issues

**"ModuleNotFoundError: No module named 'inference'"**

- Make sure you're in the `/testing` directory when running `uvicorn`
- Verify `.venv/Scripts/Activate.ps1` was run

**Server won't start**

- Check if port 8000 is already in use: `netstat -ano | findstr 8000`
- Kill the process: `taskkill /PID <PID> /F`

**Models loading slowly**

- This is normal for first run
- Models are cached; subsequent starts are faster

### Docker Issues

**"Error: failed to solve with frontend dockerfile.v0"**

- Ensure Docker Desktop is running
- Rebuild: `docker build --no-cache -t tati-geo-nli:latest -f testing/Dockerfile .`

**"Image size too large for push"**

- Consider using GitHub Container Registry (private/public)
- Or split into base image with models + smaller app layer

---

## ✅ Deployment Checklist

- [ ] Files created: `api_server.py`, `requirements_api.txt`, `Dockerfile`
- [ ] Local test passed: `curl http://localhost:8000/health` returns ok
- [ ] Docker image builds successfully
- [ ] Docker image runs locally on port 8000
- [ ] Docker image pushed to registry
- [ ] VAST AI instance created
- [ ] Endpoint accessible from VAST AI
- [ ] Sample POST request succeeds
- [ ] Model dependencies added to Dockerfile (torch, chromadb, etc.)

---

## 📞 Quick Commands Reference

```powershell
# Local setup
cd c:\Users\Vedant\Desktop\ml\tati\testing
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements_api.txt
uvicorn api_server:app --host 0.0.0.0 --port 8000

# Docker setup
cd c:\Users\Vedant\Desktop\ml\tati
docker build -t tati-geo-nli:latest -f testing/Dockerfile .
docker run --rm -p 8000:8000 tati-geo-nli:latest
docker tag tati-geo-nli:latest <username>/tati-geo-nli:latest
docker push <username>/tati-geo-nli:latest

# Test
curl http://localhost:8000/health
```

---

Done! You're ready to deploy on VAST AI. 🚀

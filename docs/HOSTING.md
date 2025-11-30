# Free Hosting Options for TEP Dashboard

This document covers free hosting alternatives for the TEP Dash web dashboard.

## Quick Comparison

| Platform | Free Tier | Spin-down on Inactivity | Recommended For |
|----------|-----------|-------------------------|-----------------|
| **Hugging Face Spaces** | Unlimited | No | Best free option |
| **Render** | 750 hrs/month | Yes (slow cold start) | Good alternative |
| **Railway** | $5 trial credit | Depletes over time | Quick testing |
| **PythonAnywhere** | 1 web app | No | Learning/simple apps |

## 1. Hugging Face Spaces (Recommended)

Hugging Face Spaces provides free Docker-based hosting with no spin-down on inactivity.

### Deployment Steps

1. **Create a Hugging Face account** at https://huggingface.co/join

2. **Create a new Space**:
   - Go to https://huggingface.co/new-space
   - Choose a name (e.g., `tennessee-eastman-process`)
   - Select **Docker** as the SDK
   - Choose **Blank** template
   - Set visibility to Public (required for free tier)

3. **Clone and push your code**:
   ```bash
   # Clone your new Space
   git clone https://huggingface.co/spaces/YOUR_USERNAME/tennessee-eastman-process
   cd tennessee-eastman-process

   # Copy files from this repository
   cp -r /path/to/tennessee-eastman-profbraatz/* .

   # Push to Hugging Face
   git add .
   git commit -m "Initial TEP dashboard deployment"
   git push
   ```

4. **Wait for build** - The Space will automatically build and deploy. Check the logs at:
   `https://huggingface.co/spaces/YOUR_USERNAME/tennessee-eastman-process`

### Key Files Used
- `Dockerfile` - Container configuration
- `requirements-dash.txt` - Python dependencies
- `tep/` - The TEP package

### Advantages
- No spin-down on inactivity (always-on)
- Free GPU upgrade available for ML models
- Built-in versioning via Git
- Easy sharing and embedding

### Resources
- [Dash on Spaces Documentation](https://huggingface.co/docs/hub/spaces-sdks-docker-dash)
- [Official Dash Template](https://github.com/plotly/dash-hugging-face-spaces)

---

## 2. Render

Render offers a generous free tier with automatic deployments from GitHub.

### Deployment Steps

1. **Create a Render account** at https://render.com

2. **Connect your GitHub repository**:
   - Click "New" → "Web Service"
   - Connect your GitHub account
   - Select the `tennessee-eastman-profbraatz` repository

3. **Configure the service**:
   - Name: `tep-dashboard`
   - Region: Choose closest to your users
   - Branch: `main`
   - Runtime: **Docker**
   - Plan: **Free**

4. **Deploy** - Click "Create Web Service"

### Alternative: Blueprint Deployment
Use the included `render.yaml` for one-click deployment:
```bash
# In the Render dashboard, click "New" → "Blueprint"
# Connect your repo - Render will auto-detect render.yaml
```

### Limitations
- **Spin-down after 15 minutes of inactivity**
- Cold start can take 30-60 seconds
- 750 free hours/month across all services

### Resources
- [Render Docker Documentation](https://render.com/docs/docker)
- [Deploy for Free](https://render.com/docs/free)

---

## 3. Railway

Railway provides a simple deployment experience with $5 free trial credit.

### Deployment Steps

1. **Create a Railway account** at https://railway.app

2. **Deploy from GitHub**:
   - Click "New Project" → "Deploy from GitHub repo"
   - Select the repository
   - Railway will auto-detect the Dockerfile

3. **Configure environment**:
   ```
   PORT=7860
   TEP_BACKEND=python
   ```

4. **Generate a domain**:
   - Go to Settings → Networking
   - Click "Generate Domain"

### Limitations
- $5 trial credit (depletes with usage)
- No persistent free tier after trial
- Good for short-term demos

### Resources
- [Railway Documentation](https://docs.railway.app/)

---

## 4. PythonAnywhere

PythonAnywhere offers a simple free tier for Python web apps.

### Deployment Steps

1. **Create account** at https://www.pythonanywhere.com

2. **Upload files**:
   - Go to "Files" tab
   - Upload the `tep/` directory and `pyproject.toml`

3. **Create a web app**:
   - Go to "Web" tab → "Add a new web app"
   - Choose "Manual configuration" → Python 3.10+
   - Set source code directory

4. **Configure WSGI**:
   Edit the WSGI configuration file:
   ```python
   import sys
   sys.path.insert(0, '/home/YOUR_USERNAME/tep-project')

   from tep.dashboard_dash import server as application
   ```

5. **Install dependencies**:
   ```bash
   pip install --user dash plotly numpy
   ```

### Limitations
- Only 1 web app on free tier
- Limited CPU time
- Outbound internet access restricted

---

## 5. Fly.io

Fly.io offers a generous free tier with global edge deployment.

### Deployment Steps

1. **Install Fly CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login and launch**:
   ```bash
   fly auth login
   fly launch --name tep-dashboard
   ```

3. **Deploy**:
   ```bash
   fly deploy
   ```

### Configuration
Create `fly.toml`:
```toml
app = "tep-dashboard"
primary_region = "iad"

[build]
  dockerfile = "Dockerfile"

[env]
  PORT = "7860"
  TEP_BACKEND = "python"

[http_service]
  internal_port = 7860
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
```

### Limitations
- Requires credit card for verification
- 3 shared-cpu VMs free
- Auto-stop after inactivity (fast restart)

---

## Local Development

For local development, use the CLI:

```bash
# Install with web dependencies
pip install -e ".[web]"

# Run the dashboard
tep-web --port 8050 --debug

# Or with Docker
docker build -t tep-dashboard .
docker run -p 7860:7860 tep-dashboard
```

## Troubleshooting

### Common Issues

1. **Port binding errors**: Ensure your Dockerfile uses `0.0.0.0` not `127.0.0.1`

2. **Memory issues**: The TEP simulator uses ~100MB RAM. Most free tiers support this.

3. **Slow startup**: First simulation step may take a few seconds to initialize.

4. **Import errors**: Make sure `gunicorn` is in `requirements-dash.txt`

### Checking Logs

- **Hugging Face**: Click "Logs" in your Space
- **Render**: Dashboard → Your Service → Logs
- **Railway**: Click on your deployment → Logs tab

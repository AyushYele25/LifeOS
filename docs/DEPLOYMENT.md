# LIFEOS — Deployment & Production Guide

This guide details how to run LIFEOS locally as a desktop web app, or deploy it publicly to cloud platforms so anyone can use it to track expenses, simulate purchases, stress-test finances, and interact with the grounded AI assistant.

---

## ⚡ Option 1: One-Click Local App (Easiest & Fastest)

To run LIFEOS locally on your computer with automatic browser opening:

```bash
# From the project root (c:\Users\ayush\LIFEOS)
python run_app.py
```

Or on Windows, simply double-click **`run.bat`**.

**What happens:**
1. Starts the FastAPI + Uvicorn server on `http://127.0.0.1:8000`.
2. Initializes SQLite tables and seeds the 28 standard Indian financial categories.
3. Automatically opens your default web browser to the modern web application.

---

## ☁️ Option 2: Free Cloud Deployment via Render.com

Render offers free web service hosting for Python web apps.

### Steps:
1. Push your LIFEOS repository to GitHub.
2. Sign up / Log in to [Render.com](https://render.com).
3. Click **New +** → **Blueprint** (or **Web Service**).
4. Connect your GitHub repository.
5. Render will automatically detect [`backend/render.yaml`](file:///c:/Users/ayush/LIFEOS/backend/render.yaml).
6. Set root directory to `backend`.
7. Click **Apply / Deploy**.

**Config Settings:**
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Health Check Path:** `/health`

Your application will be live at `https://lifeos-app.onrender.com` accessible to anyone on the web!

---

## 🚂 Option 3: Deploy via Railway.app / Heroku

Both Railway and Heroku support the included [`backend/Procfile`](file:///c:/Users/ayush/LIFEOS/backend/Procfile) and [`backend/Dockerfile`](file:///c:/Users/ayush/LIFEOS/backend/Dockerfile).

### Railway:
1. Log in to [Railway.app](https://railway.app).
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select the `LIFEOS` repository.
4. Set the root directory to `/backend` in project settings.
5. Railway will auto-detect the Dockerfile or Procfile, build dependencies, and provision a public HTTPS domain.

---

## 🐳 Option 4: Docker Container Deployment

You can build and run the production container locally or on any cloud VPS (DigitalOcean, AWS EC2, Linode):

```bash
cd backend

# Build Docker image
docker build -t lifeos-app .

# Run container on port 8000 with persistent volume for database & ChromaDB
docker run -d -p 8000:8000 -v lifeos_data:/app/data --name lifeos lifeos-app
```

Then visit `http://localhost:8000`.

---

## 🌐 Public URLs & Endpoints Summary

| Path | Purpose |
|---|---|
| **`/`** | Full modern Web Application UI (Dashboard, Expenses, Simulator, AI Agent, Goals, Risk, Obligations) |
| **`/docs`** | Interactive OpenAPI / Swagger UI documentation for all 58 REST endpoints |
| **`/health`** | Health check probe (returns `{"status": "healthy"}`) for cloud load balancers and uptime monitors |
| **`/api/auth/demo-login`** | Instant 1-click demo login (Priya Sharma) preloaded with ₹3.8L balance, salary, and obligations |
| **`/api/transactions/classify?text=...`** | Microsecond real-time ML category prediction as users type narrations |

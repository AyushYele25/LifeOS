"""
LIFEOS — One-Click Application Launcher

Launches the backend server and automatically opens the web application in your default browser.
Usage:
    python run_app.py
"""

import os
import sys
import time
import webbrowser
import subprocess
import urllib.request

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

def print_banner():
    print("\n" + "█" * 70)
    print("  🚀 STARTING LIFEOS PERSONAL LIFE OPERATING SYSTEM")
    print("  Stateful AI Financial Decision Engine for Indian Households")
    print("█" * 70 + "\n")

def check_server(url, timeout=1):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False

def main():
    print_banner()

    if not os.path.exists(BACKEND_DIR):
        print(f"❌ Error: Backend directory not found at {BACKEND_DIR}")
        sys.exit(1)

    os.chdir(BACKEND_DIR)
    host = "127.0.0.1"
    port = 8000
    app_url = f"http://{host}:{port}/"
    health_url = f"http://{host}:{port}/health"
    docs_url = f"http://{host}:{port}/docs"

    print(f"📦 Starting Uvicorn engine on {app_url}...")
    cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", host, "--port", str(port)]

    proc = subprocess.Popen(cmd)

    # Wait for server to become responsive
    print("⏳ Initializing intelligence engines, ML classifiers & SQLite database...")
    server_ready = False
    for _ in range(20):
        time.sleep(0.5)
        if check_server(health_url):
            server_ready = True
            break

    if server_ready:
        print("\n" + "═" * 70)
        print("  ✅ LIFEOS IS LIVE AND READY!")
        print("═" * 70)
        print(f"  🌐 Web Application: {app_url}")
        print(f"  📖 API Swagger UI:  {docs_url}")
        print(f"  🩺 Health Check:    {health_url}")
        print("═" * 70)
        print("  💡 Opening your web browser now... Press CTRL + C to stop server.\n")

        # Open web browser automatically
        try:
            webbrowser.open(app_url)
        except Exception:
            pass
    else:
        print("⚠️ Warning: Server is taking longer than expected to report healthy.")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n👋 Stopping LIFEOS server...")
        proc.terminate()
        proc.wait()
        print("✅ Server stopped cleanly. Good luck with your finances!")

if __name__ == "__main__":
    main()

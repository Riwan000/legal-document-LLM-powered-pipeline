"""
Helper script to start the FastAPI backend.
Run this to start the server with proper configuration.
Uses backend.config (API_HOST, API_PORT) so the server is reachable at BACKEND_URL from Streamlit.
"""
import os
import subprocess
import sys
from pathlib import Path

def main():
    """Start the FastAPI backend server."""
    from backend.config import settings
    host = os.getenv("BACKEND_HOST", settings.API_HOST)
    port = int(os.getenv("BACKEND_PORT", str(settings.API_PORT)))
    reload_enabled = os.getenv("BACKEND_RELOAD", "1").lower() in {"1", "true", "yes"}
    # 0.0.0.0 is a bind address only; browsers cannot open it. Use localhost for displayed URLs.
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print("="*70)
    print("  Starting FastAPI Backend Server")
    print("="*70)
    print(f"\nServer listening on {host}:{port}")
    print(f"Open in browser — API docs: http://{display_host}:{port}/docs")
    print("\nPress CTRL+C to stop the server\n")
    print("="*70 + "\n")
    
    try:
        # Start uvicorn server
        cmd = [
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--host", host,
            "--port", str(port),
        ]
        if reload_enabled:
            cmd += [
                "--reload",
                "--reload-dir", "backend",
            ]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
    except Exception as e:
        print(f"\n✗ Error starting server: {e}")
        print("\nMake sure you're in the project root directory.")
        print("Try running manually: uvicorn backend.main:app --reload")

if __name__ == "__main__":
    main()


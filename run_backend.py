"""
Helper script to start the FastAPI backend.
Run this to start the server with proper configuration.
"""
import os
import subprocess
import sys
from pathlib import Path

def main():
    """Start the FastAPI backend server."""
    host = os.getenv("BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("BACKEND_PORT", "8000"))
    reload_enabled = os.getenv("BACKEND_RELOAD", "1").lower() in {"1", "true", "yes"}
    print("="*70)
    print("  Starting FastAPI Backend Server")
    print("="*70)
    print(f"\nServer will start at: http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
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


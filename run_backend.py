"""
Helper script to start the FastAPI backend.
Run this to start the server with proper configuration.
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Start the FastAPI backend server."""
    print("="*70)
    print("  Starting FastAPI Backend Server")
    print("="*70)
    print("\nServer will start at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server\n")
    print("="*70 + "\n")
    
    try:
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
    except Exception as e:
        print(f"\n✗ Error starting server: {e}")
        print("\nMake sure you're in the project root directory.")
        print("Try running manually: uvicorn backend.main:app --reload")

if __name__ == "__main__":
    main()


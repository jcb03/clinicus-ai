import subprocess
import sys
import os

def main():
    """Run the FastAPI application"""
    # Change to app directory
    app_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app')
    os.chdir(app_dir)
    
    # Run FastAPI with uvicorn
    subprocess.run([
        sys.executable, "-m", "uvicorn", "api.fastapi_backend:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

if __name__ == "__main__":
    main()

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    # Change to app directory
    app_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app')
    os.chdir(app_dir)
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "main.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

if __name__ == "__main__":
    main()

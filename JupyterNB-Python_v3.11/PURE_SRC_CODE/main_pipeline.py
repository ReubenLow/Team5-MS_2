import subprocess
import os
import sys

def run_pipeline():
    # Locate the path to pipeline.sh in the temporary directory where PyInstaller extracts files
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    pipeline_script = os.path.join(base_path, "pipeline.sh")
    
    # Check if pipeline.sh exists
    if not os.path.exists(pipeline_script):
        print("pipeline.sh not found.")
        return
    
    # Run the shell script
    subprocess.call(["bash", pipeline_script])

if __name__ == "__main__":
    run_pipeline()

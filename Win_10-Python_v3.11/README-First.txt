# Windows 10 Python 3.11 Setup

This directory contains scripts and resources for setting up a Python 3.11 environment on Windows 10.

## Contents

- `install_python.bat`: Batch script to install Python 3.11.
- `setup_env.bat`: Batch script to set up the virtual environment.
- `requirements.txt`: List of Python packages required for the project.

## Instructions

1. **Install Python 3.11**:
   - Run `install_python.bat` to download and install Python 3.11.

2. **Source Virtual Environment**:
   - Execute `source myenv/Scripts/activate` to enter the virtual environment.

3. **Install the necessary packages**:
   - 'pip install -r requirements.txt` to install the necessary packages.

4. **Verify Installation**:
   - Open a command prompt.

   - Check Python version:
     ```
     python --version
     ```
   - Ensure all packages are installed:
     ```
     pip list
     ```
5. **Run pipeline**:
   - In the PURE_SRC_CODE folder:
	- Execute 'bash pipeline.sh' to start the program.


## Notes

- Ensure you have administrative privileges to install software on your system.
- Modify `requirements.txt` to add or remove packages as needed.

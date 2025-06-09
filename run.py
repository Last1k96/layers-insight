#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import argparse

def check_python():
    """Check if Python is installed."""
    try:
        subprocess.run(["python", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Python is not installed or not in PATH. Please install Python and try again.")
        return False

def create_venv():
    """Create virtual environment if it doesn't exist."""
    if not os.path.exists(".venv"):
        print("Creating virtual environment...")
        try:
            subprocess.run(["python", "-m", "venv", ".venv"], check=True)
        except subprocess.SubprocessError:
            print("Failed to create virtual environment. Please ensure you have venv module installed.")
            return False
    return True

def activate_venv():
    """Return the activation command for the virtual environment."""
    if platform.system() == "Windows":
        return os.path.join(".venv", "Scripts", "activate")
    else:
        return f"source {os.path.join('.venv', 'bin', 'activate')}"

def install_dependencies():
    """Install dependencies using pip."""
    print("Installing dependencies...")
    try:
        # Use a new process with the activated environment
        if platform.system() == "Windows":
            subprocess.run(f".venv\\Scripts\\pip install -e .", shell=True, check=True)
        else:
            subprocess.run(f".venv/bin/pip install -e .", shell=True, check=True)
    except subprocess.SubprocessError:
        print("Failed to install dependencies.")
        return False
    return True

def run_application(args):
    """Run the application with the provided arguments."""
    print("Running application...")
    cmd = []
    
    # Use python from the virtual environment
    if platform.system() == "Windows":
        cmd = [".venv\\Scripts\\python", "main.py"]
    else:
        cmd = [".venv/bin/python", "main.py"]
    
    # Add all arguments passed to this script
    cmd.extend(args)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.SubprocessError:
        print("Failed to run the application.")
        return False
    return True

def main():
    # Skip the script name
    args = sys.argv[1:]
    
    if not check_python():
        return 1
    
    if not create_venv():
        return 1
    
    if not install_dependencies():
        return 1
    
    if not run_application(args):
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
import subprocess
import sys
import os


def install_requirements():
    requirements_file = "requirements.txt"

    # Check if requirements.txt exists
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found.")
        return

    # Install packages from requirements.txt
    try:
        print(f"Installing packages from {requirements_file}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("All requirements have been successfully installed.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Installation failed with error:\n{e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    install_requirements()

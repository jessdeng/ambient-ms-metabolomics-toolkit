import subprocess
import sys
import os

def install():
    # Check requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("ERROR: requirements.txt not found.")
        print("Make sure you are running this from inside the project folder.")
        sys.exit(1)

    # Upgrade pip first to avoid install issues on fresh Python installs
    print("Updating pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install packages
    print("\nInstalling required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    print("\n[ok] Setup complete!")
    print("\nNext steps:")
    print("  1. Place your experiment folder inside  data/<EXPERIMENT_NAME>/")
    print("  2. Set EXPERIMENT in config.json to match that folder name.")
    print("  3. Run the analysis:")
    print("       python src/standard/run_analysis.py --experiment <EXPERIMENT_NAME>")


if __name__ == "__main__":
    install()
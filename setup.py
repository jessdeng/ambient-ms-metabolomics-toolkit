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
    print("  3. Run the standard pipeline:")
    print("       python -m standard.run_analysis")
    print("     Or the R-comparable pipeline:")
    print("       python -m r_comparable.run_analysis")
    print("  4. (Optional) Generate additional plots and permutation test:")
    print("       python -m standard.extras")

if __name__ == "__main__":
    install()
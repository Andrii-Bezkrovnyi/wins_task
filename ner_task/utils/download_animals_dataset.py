
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

DATASET = "alessiocorrado99/animals10"
ZIP_FILE = "animals10.zip"
EXTRACT_DIR = "animals10"


def install_kaggle():
    """Installs the Kaggle CLI if it is not installed."""
    try:
        subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
        print("[+] Kaggle CLI is already installed")
    except Exception:
        print("[*] Installing Kaggle CLI...")
        subprocess.run(["pip", "install", "kaggle"], check=True)
        print("[+] Kaggle CLI installed")


def setup_kaggle():
    """Sets up kaggle.json"""
    home = Path.home()
    kaggle_dir = home / ".kaggle"
    kaggle_json_source = Path(__file__).parent / "kaggle.json"
    kaggle_json_target = kaggle_dir / "kaggle.json"

    if not kaggle_json_source.exists() and not kaggle_json_target.exists():
        raise FileNotFoundError(
            "❌ kaggle.json file not found.\n"
            "Download it from https://www.kaggle.com/account → Create API Token\n"
            "and place it next to this script."
        )

    kaggle_dir.mkdir(exist_ok=True, mode=0o700)

    if kaggle_json_source.exists():
        shutil.move(str(kaggle_json_source), str(kaggle_json_target))
        print(f"[+] kaggle.json moved to {kaggle_json_target}")

    # Set permissions for Linux/macOS
    try:
        os.chmod(kaggle_json_target, 0o600)
    except Exception:
        pass

    print("[+] Kaggle API configured")


def download_dataset():
    if Path(ZIP_FILE).exists():
        print(f"[!] {ZIP_FILE} already exists, skipping download.")
        return
    print("[*] Downloading Animals-10 dataset...")
    subprocess.run(["kaggle", "datasets", "download", "-d", DATASET], check=True)
    print("[+] Dataset downloaded")


def unzip_dataset():
    """Extracts the archive and deletes it after extraction."""
    if Path(EXTRACT_DIR).exists():
        print(f"[!] Folder {EXTRACT_DIR} already exists, skipping extraction.")
        # If the folder exists, remove the archive if it’s still there
        if Path(ZIP_FILE).exists():
            Path(ZIP_FILE).unlink()
            print(f"{ZIP_FILE} deleted")
        return

    if not Path(ZIP_FILE).exists():
        raise FileNotFoundError(f"{ZIP_FILE} not found, please download the dataset first.")

    print(f"[*] Extracting {ZIP_FILE} → {EXTRACT_DIR} ...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print(f"[+] Dataset extracted to {EXTRACT_DIR}")

    # Delete the archive after extraction
    Path(ZIP_FILE).unlink()
    print(f"{ZIP_FILE} deleted")


def download_and_prepare_dataset():
    install_kaggle()
    setup_kaggle()
    download_dataset()
    unzip_dataset()

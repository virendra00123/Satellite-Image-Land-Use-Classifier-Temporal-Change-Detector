"""
Downloads and unpacks the two datasets used by this project into data/.
Run once, with internet access, before doing anything else:

    python data/download_data.py

After this step every other script in the project runs fully offline.
"""
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent

EUROSAT_URL = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
UCMERCED_URL = "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"


def download(url: str, dest: Path):
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
        return True
    print(f"[download] {url}")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.SSLError:
        print("[warning] SSL verification failed — retrying without verification (insecure)")
        resp = requests.get(url, stream=True, timeout=60, verify=False)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"[error] failed to download {url}: {exc}")
        return False
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    return True


def unzip(zip_path: Path, out_dir: Path):
    print(f"[unzip] {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def main():
    DATA_DIR.mkdir(exist_ok=True)

    eurosat_zip = DATA_DIR / "EuroSAT.zip"
    if download(EUROSAT_URL, eurosat_zip):
        if not (DATA_DIR / "EuroSAT").exists():
            unzip(eurosat_zip, DATA_DIR)
    else:
        print("[warning] EuroSAT not downloaded — you can place it manually under data/EuroSAT/")

    ucm_zip = DATA_DIR / "UCMerced_LandUse.zip"
    if download(UCMERCED_URL, ucm_zip):
        if not (DATA_DIR / "UCMerced_LandUse").exists():
            unzip(ucm_zip, DATA_DIR)
    else:
        print("[warning] UCMerced dataset not downloaded — you can place it manually under data/UCMerced_LandUse/Images/")

    print("Done. Directory layout:")
    for p in sorted(DATA_DIR.iterdir()):
        print(" -", p.name)

    print(
        "\nIf either mirror is down, both datasets are also mirrored on "
        "Kaggle/TensorFlow Datasets — search 'EuroSAT' and 'UC Merced Land "
        "Use Dataset' and place the extracted folders at data/EuroSAT and "
        "data/UCMerced_LandUse/Images respectively."
    )


if __name__ == "__main__":
    main()

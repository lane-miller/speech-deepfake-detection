"""
00_data/download.py

Downloads ASVspoof 5 .tar/.tar.gz files from Zenodo one at a time, straight
to the external SSD, with logging. Generates the URL list itself via
zenodo_get, so no manual copy-paste of filenames is needed.

Small text files (README, LICENSE, protocol files) are skipped — download
those manually.

Run inside a screen/tmux session for long unattended downloads.

Requires: pip install zenodo-get
"""

import subprocess
import time
import re
from pathlib import Path

DOI = "10.5281/zenodo.14498691"
DEST = Path("/Volumes/LPM03 storage/Datasets/Audio/asvspoof5")
URL_LIST_FILE = DEST / "urls.txt"
LOG_FILE = DEST / "download_log.txt"

MIN_VALID_SIZE_BYTES = 1_000_000  # 1MB — anything smaller is treated as a failed/partial download
VALID_EXTENSIONS = (".tar", ".tar.gz")
PAUSE_BETWEEN_FILES_SEC = 5  # avoid hammering Zenodo's API with rapid back-to-back calls


def log(msg):
    stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(stamped)
    with open(LOG_FILE, "a") as f:
        f.write(stamped + "\n")


def get_url_list():
    """Generate urls.txt via zenodo_get -w, without downloading anything."""
    if URL_LIST_FILE.exists():
        log(f"Using existing URL list at {URL_LIST_FILE}")
    else:
        log("Fetching URL list from Zenodo record...")
        result = subprocess.run(
            ["zenodo_get", DOI, "-w", str(URL_LIST_FILE)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            log(f"Failed to fetch URL list: {result.stderr}")
            raise SystemExit(1)
        log(f"URL list written to {URL_LIST_FILE}")

    urls = [line.strip() for line in open(URL_LIST_FILE) if line.strip()]
    return urls


def filename_from_url(url):
    # e.g. https://zenodo.org/api/records/14498691/files/flac_T_aa.tar/content
    match = re.search(r"/files/([^/]+)/content", url)
    return match.group(1) if match else url.split("/")[-1]


def is_tar_file(fname):
    return fname.endswith(VALID_EXTENSIONS)


def download_all(urls, only_pattern=None):
    DEST.mkdir(parents=True, exist_ok=True)

    tar_urls = [(url, filename_from_url(url)) for url in urls]
    tar_urls = [(url, fname) for url, fname in tar_urls if is_tar_file(fname)]

    skipped_non_tar = len(urls) - len(tar_urls)
    if skipped_non_tar:
        log(f"Skipping {skipped_non_tar} non-.tar/.tar.gz file(s) — download those manually.")

    for url, fname in tar_urls:
        if only_pattern and only_pattern not in fname:
            continue

        out_path = DEST / fname

        if out_path.exists() and out_path.stat().st_size >= MIN_VALID_SIZE_BYTES:
            log(f"Skipping {fname} (already present, {out_path.stat().st_size / 1e9:.2f} GB)")
            continue
        elif out_path.exists():
            log(f"Found incomplete/invalid {fname} ({out_path.stat().st_size} bytes) — re-downloading")
            out_path.unlink()

        log(f"Starting {fname}")
        result = subprocess.run(
            ["zenodo_get", DOI, "-g", fname, "-o", str(DEST)],
            capture_output=True, text=True
        )

        success = (
            result.returncode == 0
            and out_path.exists()
            and out_path.stat().st_size >= MIN_VALID_SIZE_BYTES
        )

        if success:
            log(f"{fname}: OK ({out_path.stat().st_size / 1e9:.2f} GB)")
        else:
            log(f"{fname}: FAILED (exit code {result.returncode})")
            if result.stdout:
                log(f"  stdout: {result.stdout.strip()}")
            if result.stderr:
                log(f"  stderr: {result.stderr.strip()}")
            if not out_path.exists():
                log(f"  -> no file was created (likely glob match failure or rate limiting)")
            elif out_path.stat().st_size < MIN_VALID_SIZE_BYTES:
                log(f"  -> file too small ({out_path.stat().st_size} bytes), possibly incomplete")

        time.sleep(PAUSE_BETWEEN_FILES_SEC)


if __name__ == "__main__":
    urls = get_url_list()
    log(f"Found {len(urls)} files in record.")

    # Set to a substring (e.g. "T_") to filter which files download first,
    # or leave as None to download everything found.
    download_all(urls, only_pattern=None)

    log("Done.")
"""
scripts/download_data.py

Downloads all OCEL log files needed for Experiment 3.
Run once before running experiments:
    python scripts/download_data.py
"""
import urllib.request
import os

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

LOGS = {
    "running_example.jsonocel": (
        "https://raw.githubusercontent.com/LukasLiss/"
        "object-centric-alignments/main/sample-logs/jsonocel/paper-example.jsonocel",
        "Paper example from Liss et al. 2023 (Package + Item)"
    ),
    "p2p_normal.jsonocel": (
        "https://raw.githubusercontent.com/ocpm/ocpa/"
        "main/sample_logs/jsonocel/p2p-normal.jsonocel",
        "Procure-to-Pay normal log from ocpa"
    ),
}

print("Downloading OCEL logs...\n")
for filename, (url, desc) in LOGS.items():
    path = os.path.join(DATA, filename)
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"  already exists: {filename}  ({size:.1f} KB)")
        continue
    print(f"  {filename}")
    print(f"  {desc}")
    try:
        urllib.request.urlretrieve(url, path)
        size = os.path.getsize(path) / 1024
        print(f"  saved → {path}  ({size:.1f} KB)\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")

print("Done.")

"""
Upload NMO benchmark data to NOMAD as a single upload + single dataset (one DOI).

Expects the flat directory produced by export_for_nomad.py:

  export_dir/
    schema.archive.yaml
    <hash>.archive.yaml   (phonon entries)
    <hash>.archive.yaml   (thermoelectric entries)
    <hash>.archive.yaml   (optomechanics entries)
    ...

All entries are zipped into one upload. The upload is wrapped in one NOMAD
dataset so the full benchmark result can be cited with a single DOI.

Authentication:
  Set the environment variable NOMAD_API_TOKEN to your personal API token.
  Generate one at https://nomad-lab.eu -> Your account -> API token.

Usage:
  export NOMAD_API_TOKEN=<your token>
  python nomad/upload_to_nomad.py \
    --export_dir   ./nomad_export/           \
    --dataset_name "NMO Benchmark Dataset"   \
    --draft
"""

import argparse
import io
import os
import sys
import time
import zipfile

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NOMAD_BASE    = "https://nomad-lab.eu/prod/v1/api/v1"
POLL_INTERVAL = 10    # seconds between status checks
POLL_TIMEOUT  = 600   # seconds before giving up


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _check(resp, context=""):
    if not resp.ok:
        print(f"\n[ERROR] {context} — HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)
    try:
        return resp.json()
    except ValueError:
        # Catches JSONDecodeError if the server returns 200 OK but sends back HTML or empty text
        print(f"\n[ERROR] {context} — Invalid JSON in response (HTTP {resp.status_code}). Response text:\n{resp.text[:500]}", file=sys.stderr)
        sys.exit(1)


def zip_directory(folder):
    """Return a BytesIO ZIP of all files directly in folder (flat, no subdirs)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, arcname=fname)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# NOMAD API calls
# ---------------------------------------------------------------------------

def upload_archive(token, upload_name, zip_buf):
    """Create the upload and push the ZIP file in a single multipart request."""
    print(f"  Uploading archive '{upload_name}' (this may take a moment) …", end=" ", flush=True)
    
    # We use files= to force a standard multipart/form-data upload. 
    # zip_buf.getvalue() extracts the raw bytes from memory.
    resp = requests.post(
        f"{NOMAD_BASE}/uploads",
        headers=_auth(token),
        params={"upload_name": upload_name},
        files={"file": ("nmo_benchmark.zip", zip_buf.getvalue(), "application/zip")},
    )
    data = _check(resp, f"upload archive '{upload_name}'")
    print("done.")
    return data["upload_id"]


def wait_for_processing(token, upload_id):
    print(f"  Processing in NOMAD backend …", end=" ", flush=True)
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        resp = requests.get(
            f"{NOMAD_BASE}/uploads/{upload_id}",
            headers=_auth(token),
        )
        info   = _check(resp, f"poll upload '{upload_id}'")
        status = info.get("data", {}).get("process_status", "")
        
        if status == "SUCCESS":
            n = info["data"].get("entries", 0)
            print(f"done ({n} entries).")
            return
        if status in ("FAILURE", "BLOCKED"):
            print(f"\n[ERROR] Processing failed with status '{status}'.", file=sys.stderr)
            sys.exit(1)
            
        time.sleep(POLL_INTERVAL)
        print(".", end="", flush=True)
        
    print(f"\n[ERROR] Timed out waiting for upload to process.", file=sys.stderr)
    sys.exit(1)


def get_entry_ids(token, upload_id):
    """Retrieve all entry IDs using pagination to safely bypass the 10k limit."""
    entry_ids = []
    page_after_value = None

    while True:
        params = {"page_size": 5000}
        if page_after_value:
            params["page_after_value"] = page_after_value

        resp = requests.get(
            f"{NOMAD_BASE}/uploads/{upload_id}/entries",
            headers=_auth(token),
            params=params,
        )
        data = _check(resp, f"list entries for upload '{upload_id}'")

        chunk = data.get("data", [])
        if not chunk:
            break

        entry_ids.extend([e["entry_id"] for e in chunk])

        pagination = data.get("pagination", {})
        page_after_value = pagination.get("next_page_after_value")

        if not page_after_value:
            break

    return entry_ids


def create_dataset(token, dataset_name):
    resp = requests.post(
        f"{NOMAD_BASE}/datasets",
        headers={**_auth(token), "Content-Type": "application/json"},
        json={"dataset_name": dataset_name, "dataset_type": "owned"},
    )
    data = _check(resp, f"create dataset '{dataset_name}'")
    return data["dataset_id"]


def assign_entries_to_dataset(token, dataset_id, entry_ids):
    if not entry_ids:
        return
    resp = requests.post(
        f"{NOMAD_BASE}/datasets/{dataset_id}/action/assign",
        headers={**_auth(token), "Content-Type": "application/json"},
        json={"entry_ids": entry_ids},
    )
    _check(resp, f"assign {len(entry_ids)} entries to dataset '{dataset_id}'")


def publish_upload(token, upload_id):
    resp = requests.post(
        f"{NOMAD_BASE}/uploads/{upload_id}/action/publish",
        headers=_auth(token),
    )
    _check(resp, f"publish upload '{upload_id}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Upload NMO benchmark export to NOMAD (single upload, single dataset)."
    )
    parser.add_argument("--export_dir",   required=True,
                        help="Flat directory produced by export_for_nomad.py.")
    parser.add_argument("--dataset_name", default="NMO Benchmark Dataset",
                        help="Name for the NOMAD dataset (receives a DOI on publication).")
    parser.add_argument("--draft",        action="store_true",
                        help="Leave upload as draft (do not publish). Recommended for review.")
    args = parser.parse_args()

    token = os.environ.get("NOMAD_API_TOKEN", "").strip()
    if not token:
        print("[ERROR] Set NOMAD_API_TOKEN environment variable.", file=sys.stderr)
        print("        Generate one at: https://nomad-lab.eu -> Your account -> API token",
              file=sys.stderr)
        sys.exit(1)

    yaml_files = [f for f in os.listdir(args.export_dir) if f.endswith(".archive.yaml")]
    if not yaml_files:
        print(f"[ERROR] No .archive.yaml files found in {args.export_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Export dir:    {args.export_dir}  ({len(yaml_files)} .archive.yaml files)")
    print(f"Dataset name:  {args.dataset_name}")
    print(f"Mode:          {'draft' if args.draft else 'publish'}\n")

    # Zip files in memory
    print("Zipping files in memory...", end=" ", flush=True)
    zip_buf = zip_directory(args.export_dir)
    print("done.")

    # Execute single-step upload
    upload_id = upload_archive(token, args.dataset_name, zip_buf)
    
    # Wait for processing
    wait_for_processing(token, upload_id)

    # Collect entry IDs (Now safely handles >10k files)
    entry_ids = get_entry_ids(token, upload_id)
    print(f"  {len(entry_ids)} entries processed and ready.")

    # Create dataset and assign all entries
    print(f"\nCreating dataset '{args.dataset_name}' …", end=" ")
    dataset_id = create_dataset(token, args.dataset_name)
    print(f"dataset_id = {dataset_id}")

    print(f"Assigning {len(entry_ids)} entries …", end=" ")
    assign_entries_to_dataset(token, dataset_id, entry_ids)
    print("done.")

    # Publish if not draft
    if not args.draft:
        print("Publishing …", end=" ")
        publish_upload(token, upload_id)
        print("done.")

    # Summary
    base_url = NOMAD_BASE.replace("/api/v1", "")
    print("\n--- Summary ---")
    print(f"  upload_id  = {upload_id}")
    print(f"  upload URL = {base_url}/user/uploads/upload/id/{upload_id}")
    print(f"  dataset_id = {dataset_id}")
    print(f"  dataset URL= {base_url}/user/datasets/dataset/id/{dataset_id}")
    if args.draft:
        print("\nUpload is in DRAFT mode. Review in the NOMAD web UI, then publish.")


if __name__ == "__main__":
    main()
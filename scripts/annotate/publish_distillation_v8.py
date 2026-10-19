"""Publish the approved archive and verify its public Hub metadata."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    DatasetCard,
    HfApi,
    get_hf_file_metadata,
    hf_hub_url,
)
from huggingface_hub.errors import RepositoryNotFoundError
import httpx


REPOSITORY = Path(__file__).resolve().parents[2]
ROOT = REPOSITORY / "outputs/sol_v8_distillation_500"
CARD = REPOSITORY / "datasets/sol_v8_distillation_500/README.md"
REPO_ID = "andaloo23/sol-v8-distillation-500"
ARCHIVE_NAME = "sol_v8_distillation_500_export_v1.zip"
RECEIPT_NAME = ARCHIVE_NAME + ".json"
STATUS_PATH = ROOT / "huggingface_upload_receipt.json"


def save_status(status):
    STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def main():
    archive = ROOT / ARCHIVE_NAME
    checksum = ROOT / RECEIPT_NAME
    card = CARD
    expected = json.loads(checksum.read_text(encoding="utf-8"))
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert archive.stat().st_size == expected["bytes"]
    assert digest == expected["sha256"]
    DatasetCard.load(str(card))

    api = HfApi()
    assert api.whoami()["name"] == REPO_ID.split("/", 1)[0]
    files = {ARCHIVE_NAME: archive, RECEIPT_NAME: checksum, "README.md": card}
    try:
        prior = api.dataset_info(REPO_ID, files_metadata=True)
    except RepositoryNotFoundError:
        prior = None
    if prior is not None:
        assert prior.private is False, "Do not alter an existing private repository"
        existing = {entry.rfilename: entry for entry in prior.siblings}
        assert set(existing) <= set(files) | {".gitattributes"}, "Unexpected existing files"
        if ARCHIVE_NAME in existing:
            assert existing[ARCHIVE_NAME].lfs.sha256 == digest, "Existing archive differs"

    repo = api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=False, exist_ok=True)
    status = {
        "repo_id": REPO_ID,
        "repo_url": str(repo),
        "status": "repository_created_upload_pending",
        "archive_bytes": expected["bytes"],
        "archive_sha256": digest,
    }
    save_status(status)
    print(json.dumps(status), flush=True)
    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                    for name, path in files.items()],
        commit_message="Publish approved Sol V8 500-episode distillation export",
    )
    status.update(status="uploaded_verification_pending", commit_sha=commit.oid,
                  commit_url=commit.commit_url)
    save_status(status)
    print(json.dumps(status), flush=True)

    public = HfApi(token=False)
    info = public.dataset_info(REPO_ID, files_metadata=True, revision=commit.oid)
    assert info.private is False
    remote_files = {entry.rfilename: entry for entry in info.siblings}
    remote_archive = remote_files[ARCHIVE_NAME]
    assert remote_archive.size == expected["bytes"]
    assert remote_archive.lfs.sha256 == digest
    url = hf_hub_url(REPO_ID, ARCHIVE_NAME, repo_type="dataset", revision=commit.oid)
    metadata = get_hf_file_metadata(url, token=False)
    assert metadata.size == expected["bytes"]
    assert metadata.etag == digest
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for name in (RECEIPT_NAME, "README.md"):
            response = client.get(hf_hub_url(REPO_ID, name, repo_type="dataset", revision=commit.oid))
            response.raise_for_status()
            assert response.content == files[name].read_bytes(), f"Readback mismatch: {name}"
    status.update(status="verified_public", public=True,
                  anonymous_metadata_verified=True, remote_sha256_verified=True,
                  readback_verified=[RECEIPT_NAME, "README.md"],
                  download_url=hf_hub_url(REPO_ID, ARCHIVE_NAME, repo_type="dataset"))
    save_status(status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Directory containing the approved ZIP and receipt")
    args = parser.parse_args()
    ROOT = args.root.resolve()
    STATUS_PATH = ROOT / "huggingface_upload_receipt.json"
    main()

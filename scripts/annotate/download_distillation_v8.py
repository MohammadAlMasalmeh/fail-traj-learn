"""Download and verify the pinned Sol/V8 release, then extract it to a fresh directory."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[2]
RELEASE_PATH = ROOT / "datasets/sol_v8_distillation_500/release.json"


def verify_archive(path, expected):
    if path.stat().st_size != expected["bytes"]:
        raise ValueError("Archive size differs from the published release")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected["sha256"]:
        raise ValueError("Archive SHA-256 differs from the published release")


def extract_archive(archive, out, directory):
    target = out / directory
    if target.exists():
        raise ValueError(f"Extraction target already exists: {target}")
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = PurePosixPath(member.filename)
            if (not path.parts or path.parts[0] != directory or path.is_absolute()
                    or ".." in path.parts or "\\" in member.filename
                    or ":" in member.filename
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=out) as temp:
            source.extractall(temp)
            (Path(temp) / directory).rename(target)
    return target


def download(out):
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    expected = release["archive"]
    out.mkdir(parents=True, exist_ok=True)
    archive = out / expected["filename"]
    if not archive.exists():
        url = (f"https://huggingface.co/datasets/{release['repo_id']}/resolve/"
               f"{release['revision']}/{expected['filename']}")
        with tempfile.NamedTemporaryFile(prefix=".download-", dir=out, delete=False) as temp:
            partial = Path(temp.name)
        try:
            print(f"Downloading pinned release {release['revision']}...", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            verify_archive(partial, expected)
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    else:
        verify_archive(archive, expected)
    print("Archive size and SHA-256 verified.", flush=True)
    target = extract_archive(archive, out, expected["directory"])
    print(f"Dataset extracted to {target}", flush=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/hf_sol_v8_distillation_500")
    args = parser.parse_args()
    download(args.out.resolve())

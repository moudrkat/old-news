#!/usr/bin/env python3
"""Push space/ to the Hugging Face Space.

    python3 space/aorus/deploy.py
    python3 space/aorus/deploy.py -m "what changed"

Uploads index.html, README.md, build_data.py and data/ only. The aorus scripts
and the promo screenshots stay out: the first are for the GPU box and the second
are for LinkedIn, and neither belongs in a page anyone can view source on.
"""
import argparse
import pathlib
import sys

REPO = "Unt1l1f1nd/turn-the-knob"
IGNORE = ["promo/*", "aorus/*", "__pycache__/*", ".DS_Store", "*.pyc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", default="Update")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    if not (root / "index.html").exists():
        sys.exit(f"no index.html under {root}")
    data = sorted((root / "data").glob("*.json"))
    if len(data) < 11:
        sys.exit(f"only {len(data)} files in data/ -- run space/build_data.py first")

    from huggingface_hub import upload_folder
    url = upload_folder(repo_id=REPO, repo_type="space", folder_path=str(root),
                        ignore_patterns=IGNORE, commit_message=args.message)
    print(url)
    print("https://huggingface.co/spaces/" + REPO)


if __name__ == "__main__":
    main()

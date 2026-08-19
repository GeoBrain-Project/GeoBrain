"""Fetch the Marmousi II sections the physics gallery reads.

The benchmark sections are 148 MB each, which is past the 100 MB ceiling a
git host will accept for a single file, so they are published as release
assets rather than committed. This script puts them where
``examples/03_physics/_models.py`` looks for them and checks each one
against a recorded digest, because a truncated download of a SEG-Y file
does not fail loudly: it reads back as a shorter section and the inversion
simply runs on the wrong earth.

Only ``vp`` is read by the gallery. ``--all`` also fetches the shear and
density sections, which ``marmousi(..., fields=("vs", "rho"))`` can return
for work of your own.

Usage::

    python examples/data/fetch_marmousi.py
    python examples/data/fetch_marmousi.py --all
    python examples/data/fetch_marmousi.py --url-base file:///mnt/benchmarks

If you already have the sections, put them in ``examples/data/marmousi/``
under the names below and run this script to verify them; nothing is
downloaded when a file is already present and correct.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent / "marmousi"

# Release assets of this project. A tag rather than `latest` so a checkout
# of 0.2.0 keeps fetching the sections 0.2.0 was measured against.
URL_BASE = "https://github.com/GeoBrain-Project/GeoBrain/releases/download/v0.2.0"

# sha256 and byte count of each section as the gallery measured it.
SECTIONS: dict[str, tuple[str, int]] = {
    "vp_marmousi-ii.segy": (
        "2722a5de0645d8fcf5750f380893d4875383af9b24988ece0bd07b943f5b0c91",
        155653444,
    ),
    "vs_marmousi-ii.segy": (
        "47713a4e03d9f0543281c6a92b91aff8ce86b388db588e2c027e67b93e5678a8",
        155653444,
    ),
    "density_marmousi-ii.segy": (
        "10ec5c4274a63ffde2ac94dda4f062d5b83827efcefeb1a2ab22438a85f32967",
        155653444,
    ),
}
REQUIRED = ("vp_marmousi-ii.segy",)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify(path: Path, expected_sha: str, expected_size: int) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size != expected_size:
        return False
    return digest(path) == expected_sha


def download(url: str, target: Path, expected_size: int) -> None:
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        read = 0
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            read += len(chunk)
            done = read / expected_size
            sys.stdout.write(f"\r    {read / 1e6:7.1f} / {expected_size / 1e6:.1f} MB"
                             f"  {'#' * int(28 * done):<28} {done * 100:5.1f}%")
            sys.stdout.flush()
    sys.stdout.write("\n")
    tmp.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the Marmousi II sections.")
    parser.add_argument("--all", action="store_true",
                        help="also fetch the shear and density sections")
    parser.add_argument("--url-base", default=URL_BASE,
                        help="where to fetch from; any URL scheme urllib "
                             "accepts, including file:// for a local copy")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    wanted = tuple(SECTIONS) if args.all else REQUIRED
    failed = []

    for name in wanted:
        expected_sha, expected_size = SECTIONS[name]
        target = DEST / name
        if verify(target, expected_sha, expected_size):
            print(f"  ok       {name}  (already present and verified)")
            continue
        url = f"{args.url_base.rstrip('/')}/{name}"
        print(f"  fetching {name}\n    from {url}")
        try:
            download(url, target, expected_size)
        except (urllib.error.URLError, OSError) as exc:
            print(f"    FAILED: {exc}")
            failed.append(name)
            continue
        if verify(target, expected_sha, expected_size):
            print(f"  ok       {name}  (verified)")
        else:
            print(f"    DIGEST MISMATCH: {name} is not the section this "
                  f"gallery was measured against; the file was kept so you "
                  f"can inspect it")
            failed.append(name)

    if failed:
        print("\nCould not obtain: " + ", ".join(failed))
        print(f"Put the files in {DEST} yourself and re-run this script to "
              f"verify them, or pass --url-base pointing at a copy you have.")
        return 1
    print(f"\nMarmousi II is ready in {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

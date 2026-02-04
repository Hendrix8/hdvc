"""
Downloader for the BIGANN / ANN_SIFT1B dataset from the TexMex corpus.

It fetches (gzipped files from the IRISA FTP server):
  - base set       (bigann_base.bvecs.gz)
  - learning set   (bigann_learn.bvecs.gz)
  - query set      (bigann_query.bvecs.gz)
  - groundtruth    (bigann_gnd.tar.gz, containing groundtruth .ivecs)

NOTE:
  The URLs and filenames below follow the standard TexMex layout that many
  public codebases use. If any URL returns 404, please visit:
      http://corpus-texmex.irisa.fr/
  and adjust the FILES/BASE_URL constants accordingly.
"""

import os
import argparse
import urllib.request
from pathlib import Path

from tqdm import tqdm


# Base FTP URL for BIGANN / ANN_SIFT1B on TexMex (IRISA FTP server)
BASE_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/"

# Remote filenames for SIFT1B / BIGANN (all gzipped on the server)
FILES = {
    "base": "bigann_base.bvecs.gz",      # ~92 GB compressed
    "learn": "bigann_learn.bvecs.gz",    # ~9.1 GB compressed
    "query": "bigann_query.bvecs.gz",    # ~964 KB compressed
    # Groundtruth distributed as tarball of ivecs files
    "groundtruth": "bigann_gnd.tar.gz",  # ~512 MB (contains .ivecs)
}


def download_file(url: str, dst_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """
    Download a file from `url` to `dst_path` with a simple progress bar.

    This does NOT support resume; if the download is interrupted you need to
    re-run the script (or manually delete the partial file).
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists():
        print(f"[SKIP] {dst_path} already exists; not re-downloading.")
        return

    print(f"[INFO] Downloading:\n  {url}\n  -> {dst_path}")

    # First request to get content length (if provided)
    try:
        with urllib.request.urlopen(url) as resp_head:
            total = int(resp_head.headers.get("Content-Length", 0))
    except Exception as e:
        print(f"[WARN] Could not get Content-Length for {url}: {e}")
        total = 0

    try:
        with urllib.request.urlopen(url) as resp, open(dst_path, "wb") as f:
            pbar = tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                desc=dst_path.name,
            )
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
            pbar.close()
    except Exception as e:
        # Clean up partial file on error
        if dst_path.exists():
            try:
                dst_path.unlink()
            except OSError:
                pass
        raise RuntimeError(f"Failed to download {url}: {e}")

    print(f"[OK] Saved to {dst_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download BIGANN / ANN_SIFT1B base/learn/query/groundtruth from TexMex."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnthdd/cpanourg/2-hdvc/data/bigann/SIFT1B",
        help="Directory to store downloaded files (default: %(default)s)",
    )
    parser.add_argument(
        "--no-base",
        action="store_true",
        default=False,
        help="Do NOT download the base set (bigann_base.bvecs).",
    )
    parser.add_argument(
        "--no-learn",
        default=False,
        action="store_true",
        help="Do NOT download the learning set (bigann_learn.bvecs).",
    )
    parser.add_argument(
        "--no-query",
        default=False,
        action="store_true",
        help="Do NOT download the query set (bigann_query.bvecs).",
    )
    parser.add_argument(
        "--no-groundtruth",
        default=True,
        action="store_true",
        help="Do NOT download the groundtruth archive (bigann_gnd.tar.gz).",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    to_download = []
    if not args.no_base:
        to_download.append("base")
    if not args.no_learn:
        to_download.append("learn")
    if not args.no_query:
        to_download.append("query")
    if not args.no_groundtruth:
        to_download.append("groundtruth")

    if not to_download:
        raise SystemExit("Nothing to download: all flags set to --no-*")

    print(f"[INFO] Downloading BIGANN / ANN_SIFT1B into: {output_dir}")
    print(f"[INFO] Will download: {', '.join(to_download)}")

    for key in to_download:
        fname = FILES[key]
        url = BASE_URL + fname
        dst = output_dir / fname
        download_file(url, dst)

    print("[DONE] All requested files processed.")


if __name__ == "__main__":
    main()


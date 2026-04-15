"""
Utilities for working with the local dataset cache.
"""

import os
import shutil
import tempfile
import json
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, Tuple, Union, IO
from hashlib import sha256

import requests
from tqdm import tqdm

CACHE_ROOT = Path(os.getenv("SCISPACY_CACHE", str(Path.home() / ".scispacy")))
DATASET_CACHE = str(CACHE_ROOT / "datasets")


def cached_path(
    url_or_filename: Union[str, Path], cache_dir: Optional[str] = None
) -> str:
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = DATASET_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError(
            "unable to parse {} as a URL or as a local path".format(url_or_filename)
        )


def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    # Only keep the file extension to stay within filesystem NAME_MAX
    # limits (e.g. 143 bytes on eCryptfs).
    _, ext = os.path.splitext(url.split("/")[-1])
    if ext:
        filename += ext

    return filename


def filename_to_url(filename: str, cache_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``FileNotFoundError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = DATASET_CACHE

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise FileNotFoundError("file {} not found".format(cache_path))

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError("file {} not found".format(meta_path))

    with open(meta_path) as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag


def http_get(url: str, temp_file: IO) -> None:
    req = requests.get(url, stream=True)
    total = int(req.headers.get("content-length", 0))
    pbar = tqdm(total=total, unit="iB", unit_scale=True, unit_divisor=1024)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            size = temp_file.write(chunk)
            pbar.update(size)
    pbar.close()


def _find_legacy_cache_path(
    url: str, etag: Optional[str], cache_dir: str
) -> Optional[str]:
    """Check for a cached file using the old naming scheme (full trailing URL component)."""
    last_part = url.split("/")[-1]
    filename = sha256(url.encode("utf-8")).hexdigest()
    if etag:
        filename += "." + sha256(etag.encode("utf-8")).hexdigest()
    filename += "." + last_part
    path = os.path.join(cache_dir, filename)
    return path if os.path.exists(path) else None


def get_from_cache(url: str, cache_dir: Optional[str] = None) -> str:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = DATASET_CACHE

    os.makedirs(cache_dir, exist_ok=True)

    response = requests.head(url, allow_redirects=True)
    if response.status_code != 200:
        raise IOError(
            "HEAD request failed for url {} with status code {}".format(
                url, response.status_code
            )
        )
    etag = response.headers.get("ETag")

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        # Check for files cached under the old naming scheme, which appended
        # the full trailing URL component instead of just the extension.
        legacy_path = _find_legacy_cache_path(url, etag, cache_dir)
        if legacy_path is not None:
            return legacy_path
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:  # type: IO
            print(f"{url} not found in cache, downloading to {temp_file.name}")

            # GET file object
            http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            print(
                f"Finished download, copying {temp_file.name} to cache at {cache_path}"
            )
            with open(cache_path, "wb") as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            meta = {"url": url, "etag": etag}
            meta_path = cache_path + ".json"
            with open(meta_path, "w") as meta_file:
                json.dump(meta, meta_file)

    return cache_path

#!/usr/bin/env python3

import os
import tarfile
import gzip
import urllib.request
import hashlib
from tqdm import tqdm


def downloadModel(link: str, outPath: str, sha256: str = None):
    """
    Downloads kaldi model files from the given link. Only links to
    the kaldi website (https://kaldi-asr.org/models) are allowed.

    Parameters
    ----------
    link : str
        Download link for the model. 
    outPath : str
        Path to output directory where the contents of the downloaded
        tarball will be extracted.
    sha256 : str, optional
        SHA-256 hash of the tarball. If provided, will check the hash
        of downloaded file against it. By default None.

    Raises
    ------
    ValueError
        If download link does not point to kaldi website.
        If hash of downloaded file does not match expected.
    IOError
        If download file size exceeds 50 MB. 
    """
    kaldiURL = "https://kaldi-asr.org/models"
    if not link.startswith(kaldiURL):
        raise ValueError(f"invalid dowload link, only models from {kaldiURL} allowed")

    if os.path.exists(outPath):
        print(f"download output path '{outPath}' is not empty, not overwriting")
        return

    maxBytes = 50 * 1024 * 1024  # 50 MB
    tarPath = f"{outPath}.tar"
    bytesRead = 0

    # Downloading and decompressing.
    pbar = tqdm(total=maxBytes, unit="bytes")
    pbar.set_description(f"downloading {link} ")

    with open(tarPath, 'wb') as tarFile:
        with urllib.request.urlopen(link) as handle:
            with gzip.GzipFile(fileobj=handle) as uncompressed:
                while bytesRead <= maxBytes:
                    data = uncompressed.read(4096)

                    if len(data) == 0:
                        pbar.update(maxBytes)
                        pbar.close()
                        break

                    bytesRead += len(data)
                    if bytesRead > maxBytes:
                        raise IOError(f"max download size ({maxBytes} bytes) exceeded")

                    pbar.update(len(data))
                    tarFile.write(data)

    # Checking hash of downloaded tarball.
    if sha256 is not None:
        with open(tarPath, 'rb') as tarFile:
            gotHash = hashlib.sha256(tarFile.read()).hexdigest()

        if sha256 != gotHash:
            os.remove(tarPath)
            raise ValueError(
                "hash of downloaded tarball does not match expected: {gotHash} vs {sha256}"
            )

    # Unzipping tarball.
    with tarfile.open(tarPath) as tarFile:
        contents = [tarinfo for tarinfo in tarFile.getmembers()]
        tarFile.extractall(os.path.dirname(outPath), members=contents)

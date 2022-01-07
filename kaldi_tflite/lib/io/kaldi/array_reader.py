#!/usr/bin/env python3

# Copyright (2021-) Shahruk Hossain <shahruk10@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import numpy as np

from kaldi_tflite.lib.io import KaldiObjReader


def ReadKaldiArray(path: str, binary: bool, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Reads either kaldi matrix or vector. This is *not* for reading archives
    (.ark) files, but rather .mat or .vec files containing a single matrix
    or vector.

    Parameters
    ----------
    path : str
        Path to matrix (.mat) or vector (.vec) file.
    binary : bool
        If true, will read the file as a binary file. Otherwise will
        parse the file as a text file.
    dtype : np.dtype, optional
        If parsing as a non binary (text) file, will type cast the read data
        into this given data type. By default np.float32.

    Returns
    -------
    np.ndarray
        Parsed matrix or vector.

    Raises
    ------
    ValueError
        If dtype is not from [np.float32, np.float64, np.int16, np.int32, np.int64].
        If the file to be parsed is not in the correct format.
    """
    # Parsing as a binary file.
    if binary:
        r = KaldiObjReader(path, True)

        # Consume header bytes for file type and then parse the specified
        # type of array.
        r.readBytes(2)
        arrayType = r.peekBytes(2).decode()

        if arrayType in ["FM", "DM", "CM"]:
            return r.readMat()
        elif arrayType in ["FV", "DV"]:
            return r.readVec()
        else:
            raise ValueError(
                f"binary file contains unexpected header bytes, {arrayType}, "
                "expected 'FV', 'DV', 'FM', 'DM' or 'CM'",
            )

    # Parsing as text file.
    # Sub routine to convert list of text tokens into list of numbers.
    def tokens2vec(tokens):
        if dtype in [np.float32, np.float64]:
            return [float(t) for t in tokens]
        elif dtype in [np.int16, np.int32, np.int64]:
            return [int(t) for t in tokens]
        else:
            raise ValueError(f"unsupported data type: {dtype}")

    mat = []
    with open(path, 'r') as f:
        for line in f:
            tokens = [t for t in line.strip().split() if t != ""]

            # Vector if opening and closing brackets on same line.
            if "[" in tokens and "]" in tokens:
                return np.array(tokens2vec(tokens[1:-1]), dtype=dtype)

            # Start of a matrix.
            if "[" in tokens:
                if len(tokens) > 1:
                    mat.append(tokens2vec(tokens[1:]))
                continue

            # End of current matrix.
            if "]" in tokens:
                if len(tokens) > 1:
                    mat.append(tokens2vec(tokens[:-1]))
                return np.array(mat, dtype=dtype)

            mat.append(tokens2vec(tokens))

    raise ValueError("reached end of file without finding closing bracket for matrix")

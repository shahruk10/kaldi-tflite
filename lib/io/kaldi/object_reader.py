#!/usr/bin/env python3

from typing import Union, Iterable
import numpy as np


class KaldiObjReader():
    """
    Reader for basic kaldi objects that can be used to construct
    readers for specific kaldi objects such as nnet3 model files.
    """

    def __init__(self, path: str, binary: bool):
        """
        Initializes the KaldiObjReader by reading in the data from the
        file at the path specified.

        Parameters
        ----------
        path : str
            Path to file.
        binary : bool
            If true, will read the file as a binary file. Otherwise will
            parse the file as a text file.
        """
        self.curPos = 0
        self.path = path
        self.binary = binary

        if not self.binary:
            raise NotImplementedError("objects in text format are currently not supported")

        self.newLineBytes = "\n".encode('utf-8')
        self.whitespaceBytes = " ".encode('utf-8')

        mode = "rb" if binary else "r"
        with open(path, mode) as fd:
            self.data = fd.read()

    def readBytes(self, nBytes: int) -> bytes:
        """
        Tries to read the specified number of bytes from the loaded data, starting
        from the current position. The read pointer is incremented by the number of
        bytes actually read.

        Parameters
        ----------
        nBytes : int
            Number of bytes to read. 

        Returns
        -------
        bytes
            List of bytes read
        """
        if self.curPos >= len(self.data):
            return []

        buf = self.data[self.curPos:self.curPos + nBytes]
        self.curPos += len(buf)

        return buf

    def expectLine(self):
        """
        Reads through the loaded data from the current position, expecting to
        find a new line character. If it does, the read pointer is incremented
        to just after the new line character. If it does not find a one, a
        ValueError is raised.

        Raises
        ------
        ValueError
            If no new line character is found between the current data index
            position and the end of the data array.
        """
        for i in range(self.curPos, len(self.data)):
            if self.data[i:i + 1] == self.newLineBytes:
                self.curPos = i + 1
                return

        raise ValueError("expected new line but did not get any")

    def readLine(self) -> str:
        """
        Reads bytes from the current position up to the next new line
        character. The read pointer is then incremented to just after the
        new line character.

        Returns
        -------
        str
            Decoded string from the read bytes.

        Raises
        ------
        ValueError
            If no new line character is found between the current read pointer
            position and the end of the data array.
        """
        for i in range(self.curPos, len(self.data)):
            if self.data[i:i + 1] == self.newLineBytes:
                line = self.data[self.curPos:i].decode()
                self.curPos = i + 1
                return line

        raise ValueError("expected new line but did not get any")

    def expectToken(self, token: str, stopTokens: Iterable[str] = []) -> bool:
        """
        Reads through the loaded data from the current position, expecting to
        find a the specificed token string. If it does, the read pointer is
        incremented to just after the end of the token string and return True.

        If stopTokens are specified, the method will return False if it happens
        to encounter any of the stop tokens before the expected token. The read
        pointer will *not* be incremented in this case.

        If the method reaches the end of the data array without encountering the
        expected token or any stop tokens, a ValueError is raised. The read
        pointer will *not* be incremented in this case.

        Parameters
        ----------
        token : str
            Token to expect.
        stopTokens : Iterable[str], optional
            List of tokens, which when encountered to should stop the search.
            By default []

        Returns
        -------
        bool
            True if it has encountered the expected token, False if not.

        Raises
        ------
        ValueError
            If expected token nor stop tokens are found. 
        """
        tokenLen = len(token)
        tokenBytes = token.encode('utf-8')

        stopTokLens = [len(t) for t in stopTokens]
        stopTokBytes = [t.encode('utf-8') for t in stopTokens]

        for i in range(self.curPos, len(self.data) - tokenLen):
            if self.data[i:i + tokenLen] == tokenBytes:
                self.curPos = i + tokenLen + 1
                return True

            if len(stopTokens) > 0:
                for l, t in zip(stopTokLens, stopTokBytes):
                    if i < len(self.data) - l:
                        if self.data[i:i + l] == t:
                            return False

        raise ValueError(f"failed to find expected token '{token}")

    def readToken(self) -> str:
        """
        Reads through the loaded data from the current position, expecting to
        find a whitespace. If it does, string between the current position and
        the position of the whitespace is returned. The read pointer is incremented
        to just after the whitespace.

        Returns
        -------
        str
            Decoded token string from the read bytes.

        Raises
        ------
        ValueError
            If no whitespace character is found between the current read pointer
            position and the end of the data array.
        """
        for i in range(self.curPos, len(self.data)):
            if self.data[i:i + 1] == self.whitespaceBytes:
                try:
                    token = self.data[self.curPos:i].decode()
                    self.curPos = i + 1
                    return token
                except UnicodeDecodeError:
                    # Decoding string may fail if the bytes between curPos and
                    # whitespace do not represent a string; this may happen in
                    # some cases where numerical values are encoded this way
                    # with whitespace delimiting.
                    continue

        raise ValueError(f"no whitespace separated token after pos {self.curPos}")

    def readBasicType(self, dtype: np.dtype) -> Union[int, float]:
        """
        Reads either an int or floating point value of the specified data type.
        The data type determines the number of bytes to read. The read pointer
        is incremented to just after the last byte of the value.

        Parameters
        ----------
        dtype : np.dtype
            Numpy data type, e.g. np.int32, np.float32, np.float64 etc. 

        Returns
        -------
        Union[int, float]
            Parsed data returned as the specified data type.

        Raises
        ------
        ValueError
            If number of bytes in the desired data type does not equal to number
            of bytes encoded for the value in the data stream.

        ValueError
            If numpy failed to parse the read bytes into the specified data type.
        """
        # Kaldi encodes data types in the form `<num-byte-specifier> <value>`;
        # that is the first byte specifies the number of bytes that should be
        # read for parsing the value. We check this byte against the number of
        # bytes in the desired data type.
        wantSize = dtype().itemsize
        gotSize = int.from_bytes(self.readBytes(1), "little")
        if gotSize != wantSize:
            raise ValueError(
                f"data type read is specified using {gotSize} bytes, but want to parse {wantSize} bytes")

        buf = self.readBytes(gotSize)
        parsed = np.frombuffer(buf, dtype=dtype)

        if len(parsed) == 0:
            raise ValueError(f"failed to parse any value of type {dtype}")

        return parsed[0]

    def readInt(self) -> int:
        """
        Reads a 32-bit integer from the data stream at the current position and
        increments the read pointer.

        Returns
        -------
        int
            Parsed integer value.
        """
        return self.readBasicType(np.int32)

    def readFloat(self) -> float:
        """
        Reads a 32-bit floating point from the data stream at the current position
        and increments the read pointer.

        Returns
        -------
        float
            Parsed float value.
        """
        return self.readBasicType(np.float32)

    def readDouble(self) -> float:
        """
        Reads a 64-bit floating point (double) from the data stream at the current
        position and increments the read pointer.

        Returns
        -------
        float
            Parsed float value.
        """
        return self.readBasicType(np.float64)

    def readBool(self) -> bool:
        """
        Reads an encoded boolean value from the data stream at the current position
        and increments the read pointer.

        Returns
        -------
        bool
            Parsed boolean value.

        Raises
        ------
        ValueError
            If the read bytes does not match expected format encoded by Kaldi for
            boolean values. 
        """
        # Bools are encoded as 8 byte chars equal to either 'T' or 'F'.
        b = self.readBytes(1)

        if b == "T".encode('utf-8'):
            return True
        elif b == "F".encode(('utf-8')):
            return False
        else:
            raise ValueError(f"unexpected format for booleans, expected 'T' or 'F', got {b}")

    def readVec(self) -> np.ndarray:
        """
        Reads a 1D array (vector) of values from the current position in the
        data stream. The data type of the elements is determined from the data
        stream by the first byte at the start of the array.

        Returns
        -------
        np.ndarray
            1D np.float32 or np.float64 array.

        Raises
        ------
        ValueError
            If the header does not contain information in the expected format
            encoded by Kaldi.
        """
        # Header containing data type.
        header = self.readBytes(3).decode()

        if header == "FV ":
            sampleSize = 4  # float
            vecType = np.float32
        elif header == "DV ":
            sampleSize = 8  # double
            vecType = np.float64
        else:
            raise ValueError(f"unknown header for vector type '{header.encode()}'")

        assert (sampleSize > 0)

        # Header containing the number of bytes to read for vector dimension.
        dimBytes = int.from_bytes(self.readBytes(1), "little")
        assert dimBytes == 4  # int32

        # Getting vector dimension.
        vecDim = np.frombuffer(self.readBytes(dimBytes), dtype=np.int32, count=1)[0]
        if vecDim == 0:
            return np.array([], dtype=vecType)

        # Read whole vector,
        buf = self.readBytes(vecDim * sampleSize)
        vec = np.frombuffer(buf, dtype=vecType)

        return vec

    def readMat(self) -> np.ndarray:
        """
        Reads a 2D array (matrix) of values from the current position in the
        data stream. The data type of the elements is determined from the data
        stream by the first byte at the start of the array.

        Returns
        -------
        np.ndarray
           2D np.float32 or np.float64 array.

        Raises
        ------
        ValueError
            If the header does not contain information in the expected format
            encoded by Kaldi.
        """
        # Header containing data type.
        header = self.readBytes(3).decode()

        if header.startswith('CM'):
            return NotImplementedError("can't decode compressed matrix yet")
        elif header == 'FM ':
            sampleSize = 4  # float
            matType = np.float32
        elif header == 'DM ':
            sampleSize = 8  # double
            matType = np.float64
        else:
            raise ValueError(f"unknown header for matrix type '{header}'")

        assert(sampleSize > 0)

        # Header containing the number of bytes to read for the number of rows.
        rowBytes = int.from_bytes(self.readBytes(1), "little")
        assert rowBytes == 4  # int32
        rows = np.frombuffer(self.readBytes(rowBytes), dtype=np.int32, count=1)[0]

        # Header containing the number of bytes to read for the number of columns.
        colBytes = int.from_bytes(self.readBytes(1), "little")
        assert colBytes == 4  # int32
        cols = np.frombuffer(self.readBytes(colBytes), dtype=np.int32, count=1)[0]

        if rows == 0 or cols == 0:
            return np.zeros((rows, cols), dtype=matType)

        # Read whole matrix.
        buf = self.readBytes(rows * cols * sampleSize)
        mat = np.frombuffer(buf, dtype=matType).reshape(rows, cols)

        return mat

#!/usr/bin/env python3

from lib.io import KaldiObjReader

class KaldiPldaReader(KaldiObjReader):
    """
    Reader for kaldi PLDA model files. It parses the model file and
    loads the parameters: mean, transformation matrix and psi.
    """

    def __init__(self, plda_path: str, binary: bool):
        """
        Initializes the KaldiPldaReader by reading in the data from the
        plda model file at the path specified.

        Parameters
        ----------
        path : str
            Path to PLDA model file.
        binary : bool
            If true, will read the file as a binary file. Otherwise will
            parse the file as a text file.
        """
        super(KaldiPldaReader, self).__init__(plda_path, binary)

        self.mean = None
        self.transformMat = None
        self.psi = None
        self.read()

    def read(self):
        """
        Reads the loaded data and parses the parameters of the
        the PLDA model.

        Raises
        ------
        ValueError
            If file is not in the expected format.
        """
        self.expectToken("<Plda>")
        self.mean = self.readVec()
        self.transformMat = self.readMat()
        self.psi = self.readVec()
        self.expectToken("</Plda>")

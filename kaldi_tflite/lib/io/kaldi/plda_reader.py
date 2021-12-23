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



from kaldi_tflite.lib.io import KaldiObjReader

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

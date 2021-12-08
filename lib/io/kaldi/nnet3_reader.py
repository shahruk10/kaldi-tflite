#!/usr/bin/env python3

from lib.io import KaldiObjReader


class KaldiNnet3Reader(KaldiObjReader):
    """
    Reader for kaldi nnet3 model files. It parses the model file and
    loads the components and their parameters.
    """

    def __init__(self, nnet3_path: str, binary: bool):
        """
        Initializes the KaldiNnet3Reader by reading in the data from the
        nnet3 model file (e.g. final.raw) at the path specified.

        Parameters
        ----------
        path : str
            Path to nnet3 model file.
        binary : bool
            If true, will read the file as a binary file. Otherwise will
            parse the file as a text file.
        """
        super(KaldiNnet3Reader, self).__init__(nnet3_path, binary)

        self.config = []
        self.components = []
        self.read()

    def read(self):
        """
        Reads the loaded data and parses the config and components of
        the nnet3 model along with their parameters. 

        Raises
        ------
        ValueError
            If file is not in the expected format.
        """
        # Token denoting the start of the nnet3 model data.
        self.expectToken("<Nnet3>")

        # Reading nnet3 config section.
        line = self.readLine()
        if line.strip() != "":
            raise ValueError("expected model config following <Nnet3> token, got blank line")
        self.readConfigLines()

        # Reading number of components in the model.
        self.expectToken("<NumComponents>")
        numComponents = self.readInt()

        assert numComponents > 0 and numComponents < 100000, \
            f"expected between 1 and 9999 components, got {numComponents}"

        # Reading the components themselves.
        self.components = []
        for i in range(numComponents):
            self.expectToken("<ComponentName>")
            # Reading component name and type.
            compName = self.readToken()
            compType = self.readToken()

            # Reading component data.
            component = {}
            component["name"] = compName
            component["type"] = compType
            component.update(self.readComponent(compType))

            # Adding to component list.
            self.components.append(component)

        self.expectToken("</Nnet3>")

    def readConfigLines(self) -> list:
        """
        Reads the lines specifying the nnet3 model config at the start
        of the model file, just after the <Nnet3> tag. The read pointer
        incremented to just after end of the config section.

        Returns
        -------
        list
            Lines containing the nnet3 model config.
        """
        self.config = []
        line = self.readLine().strip()
        while line != "":
            self.config.append(line)
            line = self.readLine().strip()

    def readComponent(self, compType: str) -> dict:
        """
        Reads the component data of the given component type. Increments
        the read pointer to just before the next component.

        Parameters
        ----------
        compType : str
            Component type to read.

        Returns
        -------
        dict
            Component data read from the data stream.
        """
        # Tokens that indicate the end of this component in the model data.
        closingTok = "</" + compType[1:]
        closingTokens = {closingTok, '<ComponentName>'}

        # Getting format of the component and the read funcs to call for each
        # of its sub parts.
        compFormat = self.getComponentFormat(compType)

        # Reading component data.
        compData = {}

        if len(compFormat) == 0:
            return compData

        for (token, readFunc, key) in compFormat:
            if self.expectToken(token, closingTokens):
                compData[key] = readFunc()
            else:
                print(f"  - failed to find token {token}")

        return compData

    def getComponentFormat(self, compType: str) -> list:
        """
        Based on the given component type, this method returns a list of
        tuples (token, read_function, dict_key) where:

          - 'token' is the name of the token to expect.

          - 'read_function' is the function we should use to read the data
            following the token.

          -'dict_key' is the key under which the read data should stored in.

        For instance, an entry in the list may be:
            ('<ParameterMatrix>', self.readMat, 'params')

        This is adapted from the following files in the kaldi repository:
          - `egs/wsj/s5/steps/nnet3/report/convert_model.py`
          - `src/nnet3/nnet-simple-component.cc`
          - `src/nnet3/nnet-normalize-component.cc`

        Parameters
        ----------
        compType : str
            Component type to get the format for.

        Raises
        ------
        ValueError
            If component type does not have a format specified.

        Returns
        -------
        list
            List of tuples (token, read_function, dict_key) corresponding
            to the given component type.
        """
        comp = self.stripTagsAndSuffix(compType, suffix="Component")

        if comp in {'Sigmoid', 'Tanh', 'RectifiedLinear', 'Softmax', 'LogSoftmax', 'NoOp'}:
            return [
                ('<Dim>', self.readInt, 'dim'),
                #  ('<BlockDim>', self.readInt, 'block-dim'),
                ('<ValueAvg>', self.readVec, 'value-avg'),
                ('<DerivAvg>', self.readVec, 'deriv-avg'),
                ('<Count>', self.readDouble, 'count'),
                ('<OderivRms>', self.readVec, 'oderiv-rms'),
                ('<OderivCount>', self.readDouble, 'oderiv-count'),
            ]

        if comp in {'Affine', 'NaturalGradientAffine'}:
            return [
                ('<LinearParams>', self.readMat, 'params'),
                ('<BiasParams>', self.readVec, 'bias'),
            ]

        if comp == 'Linear':
            return [('<Params>', self.readMat, 'params')]

        if comp == 'BatchNorm':
            return [
                ('<Dim>', self.readInt, 'dim'),
                ('<BlockDim>', self.readInt, 'block-dim'),
                ('<Epsilon>', self.readFloat, 'epsilon'),
                ('<TargetRms>', self.readFloat, 'target-rms'),
                ('<TestMode>', self.readBool, 'test-mode'),
                ('<Count>', self.readDouble, 'count'),
                ('<StatsMean>', self.readVec, 'stats-mean'),
                ('<StatsVar>', self.readVec, 'stats-var'),
            ]

        # We don't parse any parameters for these components.
        if comp in {"StatisticsExtraction", "StatisticsPooling"}:
            return []

        raise ValueError(f"unsupported component type '{compType}'")

    def stripTagsAndSuffix(self, token: str, suffix: str = "") -> str:
        """
        Strips angle brackets and any specified suffix from the given
        token.

        Parameters
        ----------
        token : str
            Token to strip angle bracket tags and suffix from.
        suffix : str, optional
            Suffix to strip from the token, by default ""

        Returns
        -------
        str
            Cleaned up token.
        """
        # Striping angle brackets.
        if token.startswith("<"):
            token = token.lstrip("<")
        if token.endswith("/>"):
            token = token.rstrip("/>")
        if token.endswith(">"):
            token = token.rstrip(">")

        # Stripping any additional suffixes.
        if len(suffix) > 0:
            if token.endswith(suffix):
                token = token[:len(token) - len(suffix)]

        return token

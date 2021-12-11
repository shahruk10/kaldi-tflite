#!/usr/bin/env python3

import os

from lib.testdata import KaldiTestDataReader


class RefKaldiXVectorModels(KaldiTestDataReader):
    basePath = os.path.dirname(__file__)

#!/usr/bin/env python3

import os

from lib.testdata import KaldiTestDataReader


class RefStatsPooling(KaldiTestDataReader):
    basePath = os.path.dirname(__file__)

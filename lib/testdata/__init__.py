#!/usr/bin/env python3

from .testdata import KaldiTestDataReader

from .plda.plda_model import RefPldaModel
from .plda.plda_scores import RefPldaScores

from .xvectors.xvectors import RefXVectors

from .tdnn.tdnn_narrow import RefTdnnNarrow
from .tdnn.tdnn_single_layer import RefTdnnSingleLayer
from .tdnn.tdnn_models import RefKaldiXVectorModels

from .stats.stats_pooling import RefStatsPooling

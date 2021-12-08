#!/usr/bin/env python3

import numpy as np

def reshapeKaldiTdnnWeights(weights: np.ndarray, units: int, kernel_width: int) -> np.ndarray:
    """
    Reshapes Kaldi's 2D TDNN parameter matrix into a shape that
    can be used to initialize the time kernel of the TDNN layer
    defined in lib.layers.
    """
    return weights.flatten().reshape((1, -1, kernel_width, units), order="F").transpose([0,2,1,3])

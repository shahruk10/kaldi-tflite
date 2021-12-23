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



from typing import Iterable

from tensorflow.keras.layers import Layer, Input, ReLU
from tensorflow.keras.models import Sequential

from kaldi_tflite.lib.layers import TDNN, BatchNorm, StatsPooling
from kaldi_tflite.lib.io import KaldiNnet3Reader


def cfg2layers(layerCfg: dict) -> Iterable[Layer]:
    """
    Uses the given layer config to instantiate one or more
    tensorflow layers.

    Parameters
    ----------
    layerCfg : dict
        Layer config. May specify multiple layers by specifying
        a list of layer types, e.g. ["affine", "relu", "batchnorm"].
        See ${TOP}/data/kaldi_models/configs/ for examples.

    Returns
    -------
    Iterable[Layer]
        One or more layers instantiated from the layer config.

    Raises
    ------
    KeyError
        If the layer config is missing necessary properties.

    ValueError
        If the type specified in the layer config is not supported.
    """

    layerTypes = layerCfg.get("type", [])
    if isinstance(layerTypes, str):
        layerTypes = [layerTypes]
    if len(layerTypes) == 0:
        raise KeyError("layer config does not define layer 'type'")

    name = layerCfg.get("name", None)

    layers = []
    for layerType in layerTypes:
        t = layerType.lower()
        cfg = layerCfg.get("cfg", {})

        if t in ["affine", "tdnn"]:
            cfg["name"] = f"{name}.affine"
            layer = TDNN(**cfg)
        elif t in ["relu"]:
            layer = ReLU(name=f"{name}.relu")
        elif t in ["batchnorm", "bn"]:
            layer = BatchNorm(name=f"{name}.batchnorm")
        elif t in ["stats", "stats_extraction", "stats_pooling"]:
            cfg["name"] = name
            layer = StatsPooling(**cfg)
        else:
            raise ValueError(f"unsupported layer type '{t}'")

        layers.append(layer)

    return layers


def SequentialFromConfig(cfg: dict, nnet3Path: str = None) -> Sequential:
    """
    Creates a tensorflow.keras.Sequential model using the given configuration.

    Parameters
    ----------
    cfg : dict
        Model config. See ${TOP}/data/kaldi_models/configs/ for examples.

    Returns
    -------
    Sequential
        Model created using the config.

    nnet3Path : str, optional
        If path to nnet3 raw file provided, the created tensorflow model
        will be initialized using weights from nnet3 model. By default None.

    Raises
    ------
    ValueError
        If config is missing layer configs.
        If first layer in layer config is not an input layer.
    """
    layersConfig = cfg.get("layers", [])
    if len(layersConfig) == 0:
        raise ValueError("no layers defined in config")

    layers = []

    # First layer should be input.
    inputCfg = layersConfig[0]
    if inputCfg.get("type", "") != "input":
        raise ValueError("first layer in sequential model needs to be of type 'input'")

    batchSize, timesteps, featDim = inputCfg["shape"]
    layers.append(Input(shape=(timesteps, featDim), batch_size=batchSize))

    # Creating rest of the layers.
    for lCfg in cfg["layers"][1:]:
        layers.extend(cfg2layers(lCfg))

    mdl = Sequential(layers)

    # Initializing weights if path to nnet3 model given.
    if nnet3Path is not None:
        nnet3Mdl = KaldiNnet3Reader(nnet3Path, True)
        for layer in mdl.layers:
            try:
                layer.set_weights(nnet3Mdl.getWeights(layer.name))
            except KeyError:
                print(f"component with name '{layer.name}' not found in nnet3 model, "
                      "skipping initialization")

    return mdl

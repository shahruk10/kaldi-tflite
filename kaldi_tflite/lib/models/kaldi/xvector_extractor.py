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

import os
import yaml

import tensorflow as tf
import kaldi_tflite as ktf


def XvectorExtractorFromConfig(cfgPath: str, name: str = None) -> tf.keras.Model:
    """
    Creates a kaldi x-vector extractor model which contains the feature
    extraction layers as well as the TDNN model. It will also initialize the
    model weights using the Kaldi model downloaded from the Kaldi website.

    See `data/tflite_models` for example configs.

    Parameters
    ----------
    cfgPath : str
        Path to config file specifying layer configs, and model weight paths.
        See `data/tflite_models` for example configs.
    name : str, optional
        Name of the model for reference, by default None
    Returns
    -------
    tf.keras.Model
        Built and initialized x-vector extractor model.
    """
    # Loading config file.
    with open(cfgPath) as f:
        cfg = yaml.safe_load(f)

    # Loading the config for the corresponding kaldi x-vector model.
    with open(cfg["extractor"]["xvec"]["model_config_path"], 'r') as f:
        kaldiCfg = yaml.safe_load(f)

    # Checking if the kaldi model exists. If not will download it from
    # the kaldi website.
    kaldiMdlPath = cfg["extractor"]["xvec"]["model_path"]
    if not os.path.exists(kaldiMdlPath):
        downloadDir = os.path.join(
            os.path.dirname(cfg["extractor"]["xvec"]["model_config_path"]),
            kaldiCfg["name"],
        )

        link = kaldiCfg["download"]["link"]
        tarHash = kaldiCfg["download"]["hash"]

        ktf.models.kaldi.downloadModel(link, downloadDir, tarHash)

    # Creating extractor.
    extractor = XvectorExtractor(cfg["extractor"], name=name)
    extractor(tf.keras.Input((None,), dtype=tf.float32))

    return extractor


class XvectorExtractor(tf.keras.Model):

    """
    Tensorflow model that accepts audio samples of an arbitrary length and outputs the
    x-vector computed from the audio. The x-vector returned is also "whitened" by applying
    a LDA transformation and length normalized, ready to be scored by a PLDA model.

    The model comprises the following components:
      - Audio sample framing
      - Windowing
      - MFCC feature extraction
      - Voice activity detection
      - Sliding window CMVN
      - X-vector computation
      - X-vector whitening (via LDA) and length normalization

    The expected input shape is (N,) where N is the number of audio samples. The audio
    samples must be float32 values between -32767.0 and +32767.0

    The output shape is (dim, ) where dim is the dimensions of the output x-vector.
    """

    def __init__(self, cfg: dict, name: str = None, chunk_size: int = 300, **kwargs):
        """
        Creates and initializes the weights of a Kaldi x-vector extraction model.

        Parameters
        ----------
        cfg : dict
            Dictionary containign configs for the `Framing` layer, `MFCC` layer,
            `VAD` layer, `CMVN` layer and the x-vector TDNN sequential model.  
        name : str, optional
            Name of the model for reference, by default None
        """
        super(XvectorExtractor, self).__init__(name=name, **kwargs)

        # Creating feature extraction layers.
        self.framing = ktf.layers.Framing(**cfg["framing"])
        self.mfcc = ktf.layers.MFCC(**cfg["mfcc"])
        self.vad = ktf.layers.VAD(**cfg["vad"])
        self.cmvn = ktf.layers.CMVN(**cfg["cmvn"])

        # Initializing x-vector model.
        with open(cfg["xvec"]["model_config_path"], 'r') as f:
            nnet3Cfg = yaml.safe_load(f)
        self.xvec = ktf.models.SequentialFromConfig(
            nnet3Cfg["model_config"], cfg["xvec"]["model_path"], "cmvn2xvec",
        )

        # Loading global mean x-vector that's substracted from the computed
        # x-vector before applying LDA.
        globalMean = ktf.io.ReadKaldiArray(cfg["xvec"]["global_mean_path"], binary=False)
        self.xvecGlobalMean = tf.constant(globalMean, dtype=self.xvec.dtype)

        # Loading LDA transform matrix.
        ldaMat = ktf.io.ReadKaldiArray(cfg["xvec"]["lda_matrix_path"], binary=True)

        # The LDA matrix file contains the transform matrix plus the mean offset
        # terms in the last column.
        self.ldaOffset = tf.constant(ldaMat[..., -1:].T, dtype=tf.float32)   # 1 x lda_dim
        self.ldaMat = tf.constant(ldaMat[..., :-1].T, dtype=tf.float32)      # xvec_dim x lda_dim

    @tf.function
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Accepts a tf.float32 tensor containing audio waveform samples
        scaled to be between +/- 32767.0 (int16 max) and returns 
        x-vectors extracted from the audio, and transformed via LDA.

        Parameters
        ----------
        inputs : tf.Tensor, shape = (batch, samples)
            tf.float32 tensor containing audio waveform samples
            scaled to be between +/- 32767.0 (int16 max) and returns 
            x-vectors extracted from the audio, and transformed via LDA.
        training : bool, optional
            If True, will cause trainable layers to be updated when
            backpropgating and dropout layers to become active during
            inference. By default False. 
        Returns
        -------
        tf.Tensor, shape = (batch, xvec_lda_dim)
            Extracted x-vectors, and transformed using LDA.
        """
        # Framing audio samples and extracting MFCCs.
        x = self.framing(inputs)
        x = self.mfcc(x)

        # Computing VAD.
        activeFrames = self.vad(x)
        x = tf.gather_nd(x, activeFrames)
        x = tf.expand_dims(x, 0)

        # Computing CMVN.
        x = self.cmvn(x)

        # Computing x-vector.
        x = self.xvec(x, training=training)

        # Subtracting global mean and applying LDA.
        x = x - self.xvecGlobalMean
        x = tf.matmul(x, self.ldaMat) + self.ldaOffset

        # Length normalizing resulting vector.
        norm = tf.norm(x, ord=2, axis=-1, keepdims=True)
        dim = tf.cast(tf.shape(x)[-1], x.dtype)
        ratio = tf.divide(norm, tf.sqrt(dim))
        x = tf.divide(x, ratio)

        # Removing redundant batch and frame axes.
        x = tf.squeeze(x)

        return x

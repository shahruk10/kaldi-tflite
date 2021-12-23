# Kaldi Models in Tensorflow Lite

[![CI](https://github.com/shahruk10/kaldi-tflite/actions/workflows/ci.yml/badge.svg)](https://github.com/shahruk10/kaldi-tflite/actions/workflows/ci.yml)

- This repo contains tensorflow python code defining components in the typical Kaldi pipelines, such as those involving x-vector models for speaker ID and diarization.

- The components are defined as tensorflow [`Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) or [`Layer`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) classes using regular tensorflow ops which can then easily be assembled and converted to Tensorflow Lite model format for inference.

- This project still a **work in progress**. I hope to add more examples and expand the feature set soon.


## Installation

```sh
# Doing this in a virtual environment is recommended. 
git clone https://github.com/shahruk10/kaldi-tflite
cd kaldi-tfilite && pip install -e .
```

## Implemented Layers

- Layer implementations can be found under [`lib/layers`](./lib/layers). The implementations are kaldi compatible, i.e. they produce the same output (within numerical rounding tolerances) as the Kaldi counterparts.

- List of implemented layers (**all compatible with TF Lite**) include:

  - [`Framing`](./lib/layers/dsp/framing.py)
  - [`Windowing`](./lib/layers/dsp/windowing.py)
  - [`FilterBank`](./lib/layers/dsp/filterbank.py)
  - [`DCT`](./lib/layers/dsp/dct.py)
  - [`MFCC`](./lib/layers/dsp/mfcc.py)
  - [`CMVN`](./lib/layers/normalization/cmvn.py)
  - [`TDNN`](./lib/layers/tdnn/tdnn.py)
  - [`StatsPooling`](./lib/layers/stats/stats_pooling.py)
  - [`PLDA`](./lib/layers/plda/plda.py)

- `TDNN` and `BatchNorm` Layers can be easily initialized from existing Kaldi nnet3 model files. See [`SequentialFromConfig`](./lib/model/kaldi/sequential.py) and [`data/kaldi_model/configs`](./data/kaldi_model/configs) for examples of how Kaldi's pret-rained x-vector models can be converted to Tensorflow Lite models.


## Usage

- TODO: organize and expand this.

### Creating a feature extraction model

```py
#!/usr/bin/env python3

import tensorflow as tf
import kaldi_tflite as ktf

# Defining MFCC feature extractor + CMVN. Input = raw audio samples (between +/-32767).
mfcc = tf.keras.models.Sequential([
    tf.keras.layers.Input((None,)),
    ktf.layers.Framing(dynamic_input_shape=True),
    ktf.layers.MFCC(num_mfccs=30, num_mels=30),
    ktf.layers.CMVN(center=True, window=200, norm_vars=False),
], name="wav2mfcc")

print(mfcc.summary())
```

### Creating a TDNN model

- Creating a trainable TDNN model for extracting x-vectors following the architecture used in
Kaldi recipes [like this one](https://github.com/kaldi-asr/kaldi/blob/054af6bda820a96dd8d026d144a5263314f31dd3/egs/sitw/v1/local/nnet3/xvector/tuning/run_xvector_1a.sh#L94):

```py
# Defining x-vector model. Input = MFCC frames with 30 mfcc coefficients.
xvec = tf.keras.models.Sequential([
    tf.keras.layers.Input((None, 30)),
    ktf.layers.TDNN(512, context=[-2, -1, 0, 1, 2], activation="relu", name="tdnn1.affine"),
    ktf.layers.BatchNorm(name="tdnn1.batchnorm"),
    ktf.layers.TDNN(512, context=[-2, 0, 2], activation="relu", name="tdnn2.affine"),
    ktf.layers.BatchNorm(name="tdnn2.batchnorm"),
    ktf.layers.TDNN(512, context=[-3, 0, 3], activation="relu", name="tdnn3.affine"),
    ktf.layers.BatchNorm(name="tdnn3.batchnorm"),
    ktf.layers.TDNN(512, context=[0], activation="relu", name="tdnn4.affine"),
    ktf.layers.BatchNorm(name="tdnn4.batchnorm"),
    ktf.layers.TDNN(1500, context=[0], activation="relu", name="tdnn5.affine"),
    ktf.layers.BatchNorm(name="tdnn5.batchnorm"),
    ktf.layers.StatsPooling(left_context=0, right_context=10000, reduce_time_axis=True),
    ktf.layers.TDNN(512, context=[0], name="tdnn6.affine"),
], name="mfcc2xvec")

print(xvec.summary())
```

### Initializing model weights from a Kaldi nnet3 model file

```py
# Loading model weights from kaldi nnet3 file. For x-vector models,
# only TDNN and BatchNorm layers need to be initialized. We match
# the layers with components in the nnet3 model using names.
nnet3Path = "path/to/final.raw"
nnet3 = ktf.io.KaldiNnet3Reader(nnet3Path, True)
for layer in xvec.layers:
    try:
        layer.set_weights(nnet3.getWeights(layer.name))
    except KeyError:
        print(f"'{layer.name}' not found in nnet3 model, skipping")
```

### Combining feature extraction with neural network

- Feature extraction layers or models can be combined with other tensorflow layers
to create a single model.

```py
# Combined feature extraction and x-vector models.
mdl = tf.keras.models.Sequential([
    tf.keras.layers.Input((None,)),
    mfcc,
    xvec,
], name="wav2xvec")

print(mdl.summary())
```

### Saving and Converting to a Tensorflow Lite model

- To convert the model to a tensorflow lite model, we first need to save the model
to disk as a `SavedModel`. Then we can use the `SavedModel2TFlite` method to convert
and optimize the model for Tensorflow Lite.

```py
savedMdl = "path/to/saved-model-directory"
tfliteMdl = "path/to/tflite-model-file"

mdl.save(savedMdl)
ktf.models.SavedModel2TFLite(savedMdl, tfliteMdl, optimize=True)
```

## Related Projects

- There are a couple of related projects out there that served as very useful reference, especially for the feature extraction layers:

  - [kaldifeat](https://github.com/yuyq96/kaldifeat): Kaldi feature extraction implemented in NumPy
  - [torchaudio](https://github.com/pytorch/audio): Kaldi feature extraction implemented in PyTorch
  - [zig-audio](https://github.com/happyalu/zig-audio): SPTK feature extraction implemented in Zig

## License

[Apache License 2.0](LICENSE)

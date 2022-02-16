# Kaldi Models in Tensorflow Lite

[![CI](https://github.com/shahruk10/kaldi-tflite/actions/workflows/ci.yml/badge.svg)](https://github.com/shahruk10/kaldi-tflite/actions/workflows/ci.yml)

- This repo contains tensorflow python code defining components in the typical Kaldi pipelines, such as those involving x-vector models for speaker ID and diarization.

- The components are defined as tensorflow [`Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) or [`Layer`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) classes using regular tensorflow ops which can then easily be assembled and converted to Tensorflow Lite model format for inference.

- This project still a **work in progress**. I hope to add more examples and expand the feature set soon.

## Installation

```sh
# Doing this in a virtual environment is recommended. 
git clone https://github.com/shahruk10/kaldi-tflite
cd kaldi-tflite && pip install -e .
```

## Implemented Layers

- Layer implementations can be found under [`kaldi_tflite/lib/layers`](./kaldi_tflite/lib/layers). The implementations are kaldi compatible, i.e. they produce the same output (within numerical rounding tolerances) as the Kaldi counterparts.

- List of implemented layers (**all compatible with TF Lite**) include:

  - [`Framing`](./kaldi_tflite/lib/layers/dsp/framing.py)
  - [`Windowing`](./kaldi_tflite/lib/layers/dsp/windowing.py)
  - [`FilterBank`](./kaldi_tflite/lib/layers/dsp/filterbank.py)
  - [`DCT`](./kaldi_tflite/lib/layers/dsp/dct.py)
  - [`MFCC`](./kaldi_tflite/lib/layers/dsp/mfcc.py)
  - [`VAD`](./kaldi_tflite/lib/layers/dsp/vad.py)
  - [`CMVN`](./kaldi_tflite/lib/layers/normalization/cmvn.py)
  - [`TDNN`](./kaldi_tflite/lib/layers/tdnn/tdnn.py)
  - [`StatsPooling`](./kaldi_tflite/lib/layers/stats/stats_pooling.py)
  - [`PLDA`](./kaldi_tflite/lib/layers/plda/plda.py)

- `TDNN` and `BatchNorm` Layers can be easily initialized from existing Kaldi nnet3 model files.
See [`SequentialFromConfig`](./kaldi_tflite/lib/models/kaldi/sequential.py) and [`data/kaldi_models/configs`](./data/kaldi_models/configs) for examples of how Kaldi's pret-rained x-vector models can be converted to Tensorflow Lite models.

## Usage

- TODO: organize and expand this.

### Quickstart

- To build and initialize an x-vector extractor model using pre-trained weights:

```py
#!/usr/bin/env python3

import librosa
import numpy as np

import tensorflow as tf
import kaldi_tflite as ktf

# Config file specifying feature extracting configs, nnet3 model config and weight paths. 
cfgPath = "data/tflite_models/0008_sitw_v2_1a.yml"

# Building and initializing model; Input = raw audio samples (between +/-32767).
# Output = x-vector with LDA and length normalization applied.
mdl = ktf.models.XvectorExtractorFromConfig(cfgPath, name="wav2xvec")
print(mdl.summary())


# Loading a test audio file; librosa converts samples to be within +/- 1.0; but
# Kaldi and this library expects +/- 32767.0 (i.e. int16 max).
inputFile = "kaldi_tflite/lib/testdata/librispeech_2.wav"
samples, _ = librosa.load(inputFile, sr=None)
samples = np.float32(samples * 32767.0)

# Adding a batch axis.
samples = samples.reshape(1, -1)

# Extract x-vectors.
xvec = mdl(samples)
print(xvec)
```

- The sections below show a more detailed step-by-step process of building models in different
ways using the layers and functionality implemented in this repo.

### Creating a feature extraction model

- We can use the traditional ways of defining a tensorflow / keras model that extract features from provided raw audio samples using the layers defined in the module. The example uses the `Sequential` model structure from keras, but we can easily use `Functional` or `sub-classed` models for more flexibility.
  
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

- To filter silent frames, we can use the [`VAD`](./kaldi_tflite/lib/layers/dsp/vad.py) layer in a functional model.

- **WARNING**: Using the default layer config, the `VAD` can be used with a batch size of 1 only, since the number of 'active' frames in different entries in a batch won't necessarily be the same. The workaround for this is setting `return_indexes=False` in the layer config, which will cause it to output a *binary mask* instead (active frames = 1, silent frames = 0), which can be used as necessary.

```py
#!/usr/bin/env python3

import tensorflow as tf
import kaldi_tflite as ktf

# Defining MFCC feature extractor + VAD + CMVN. Input = raw audio samples (between +/-32767).

# Input layer.
i = tf.keras.layers.Input((None,))

# DSP layers.
framingLayer = ktf.layers.Framing(dynamic_input_shape=True)
mfccLayer = ktf.layers.MFCC(num_mfccs=30, num_mels=30)
cmvnLayer = ktf.layers.CMVN(center=True, window=200, norm_vars=False)
vadLayer = ktf.layers.VAD()

# Creating frames and computing MFCCs.
x = framingLayer(i)
x = mfccLayer(x)

# Computing VAD, which returns indexes of 'active' frames by default. But it can be made to
# output a mask instead as well. See layer configs.
activeFrames = vadLayer(x)
x = tf.gather_nd(x, activeFrames)

# The `tf.gather_nd` above removes the batch dimension, so we add one back.
x = tf.expand_dims(x, 0)

# Computing CMVN
x = cmvnLayer(x)

mfcc = tf.keras.Model(inputs=i, outputs=x, name="wav2mfcc")

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

- Models created using this library can be initialized using weights from Kaldi nnet3 model files. For the TDNN model in the example [above](#creating-a-tdnn-model), you can download it from the [Kaldi website](https://kaldi-asr.org/models/m8) (SITW Xvector System 1a). Then we can load the nnet3 file and initialize our tensorflow model.

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

- For applications involving long audio streams, it is recommended to keep the feature extraction
and neural networks separate as models to facililate processing chunks of frames at a time. The two
tensorflow lite models can be made to work together with some glue code to pad and stream frames to
each as required.

### Saving and Converting to a Tensorflow Lite model

- To convert the model to a tensorflow lite model, we first need to save the model
to disk as a `SavedModel`. Then we can use the `SavedModel2TFlite` method to convert
and optimize the model for Tensorflow Lite.

```py
# Set the paths where the models will be saved.
savedMdl = "./my-model"
tfliteMdl = "./my-model.tflite"

# If you want to optimize the model for embedded devices, set this to True. This will
# cause the tensorflow lite model to be optimized for size and latency, as well as
# utilize use ARM's Neon extensions for math computations. (This may cause the model
# to actually run *slower* on x86 systems that don't use Neon).
optimize=False
targetDtypes = [ tf.float32 ]

mdl.save(savedMdl)
ktf.models.SavedModel2TFLite(savedMdl, tfliteMdl, optimize=optimize, target_dtypes=targetDtypes)
```

## Related Projects

- There are a couple of related projects out there that served as very useful reference, especially for the feature extraction layers:

  - [kaldifeat](https://github.com/yuyq96/kaldifeat): Kaldi feature extraction implemented in NumPy
  - [torchaudio](https://github.com/pytorch/audio): Kaldi feature extraction implemented in PyTorch
  - [zig-audio](https://github.com/happyalu/zig-audio): SPTK feature extraction implemented in Zig

## License

[Apache License 2.0](LICENSE)

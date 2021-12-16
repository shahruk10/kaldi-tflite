# Kaldi Models in Tensorflow Lite

[![CI](https://github.com/shahruk10/kaldi-tflite/actions/workflows/ci.yml/badge.svg)](https://github.com/shahruk10/kaldi-tflite/actions/workflows/ci.yml)

- This repo contains tensorflow python code defining components in the typical Kaldi pipelines, such as those involving x-vector models for speaker ID and diarization.

- The components are defined as tensorflow [`Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) or [`Layer`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) classes using regular tensorflow ops which can then easily be assembled and converted to Tensorflow Lite model format for inference.

- This project still a work in progress. I hope to add more examples and expand the feature set soon.

## Implemented Layers

- Layer implementations can be found under [`lib/layers`](./lib/layers). The implementations are kaldi compatible, i.e. they produce the same output (within numerical rounding tolerances) as the Kaldi counterparts.

- List of implemented layers (**all compatible with TF Lite**) include:

  - [`Framing`](./lib/layers/dsp/framing.py)
  - [`Windowing`](./lib/layers/dsp/windowing.py)
  - [`FilterBank`](./lib/layers/dsp/filterbank.py)
  - [`DCT`](./lib/layers/dsp/dct.py)
  - [`MFCC`](./lib/layers/dsp/mfcc.py)
  - [`TDNN`](./lib/layers/tdnn/tdnn.py)
  - [`StatsPooling`](./lib/layers/stats/stats_pooling.py)
  - [`PLDA`](./lib/layers/plda/plda.py)

- `TDNN` and `BatchNorm` Layers can be easily initialized from existing Kaldi nnet3 model files. See [`lib/model/sequential.py`](./lib/model/sequential.py) and [`data/configs`](./data/configs) for examples of how Kaldi's pret-rained x-vector models can be converted to Tensorflow Lite models.

## Related Projects

- There are a couple of related projects out there that served as very useful reference, especially for the feature extraction layers:

  - [kaldifeat](https://github.com/yuyq96/kaldifeat): Kaldi feature extraction implemented in NumPy
  - [torchaudio](https://github.com/pytorch/audio): Kaldi feature extraction implemented in PyTorch
  - [zig-audio](https://github.com/happyalu/zig-audio): SPTK feature extraction implemented in Zig

## License

[Apache License 2.0](LICENSE)

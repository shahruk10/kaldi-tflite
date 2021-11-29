# Kaldi X-Vector Diariazation Pipeline in Tensorflow Lite

- This repo contains tensorflow python code defining components in the typical Kaldi diarization pipelines involving x-vector models.

- The components are defined as tensorflow [`Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) or [`Layer`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) classes using regular tensorflow ops which can then easily be assembled and converted to Tensorflow Lite model format for inference.

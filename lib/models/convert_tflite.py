#!/usr/bin/env python3

import os
from pprint import pformat

import tensorflow as tf


def SavedModel2TFLite(
    savedModelPath: str, outPath: str,
    optimize: bool = True, enable_select_tf_ops: bool = False,
):
    """
    Converts a SavedModel into a TFLite model saved as a FlatBuffer
    file. Not all models may be convertible. Please check Tensorflow
    docs for all supported model layers and operations:
        https://www.tensorflow.org/lite/convert 
    Parameters
    ----------
    savedModelPath : str
        Path to SavedModel directory.
    outPath : str
        Path to where converted TFLite model will be written
        (.tflite extension included if not provided)
    optimize : bool, optional
        Applies latency and model size optimizations, such as
        dynamic quantization of model weights, by default True
    enable_select_tf_ops: bool, optional
        If true, will allow ops from the core tensorflow library which will
        require linking to the flex ops library when building applications
        with the TF Lite API.
    """

    print(f"TFLiteConverter: using tensorflow v{tf.__version__}")

    converter = tf.lite.TFLiteConverter.from_saved_model(savedModelPath)
    if enable_select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
    else:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        ]

    # Setting optimizations for model size and latency
    if optimize:
        print(f"Optimizing for model size and inference latency")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    else:
        print(f"*Not* doing any optimizations for model size and inference latency")
        converter.optimizations = []

    converter.experimental_new_converter = True
    tfliteModel = converter.convert()

    # Save the model
    outPath = os.path.abspath(outPath)
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    if not outPath.endswith(".tflite"):
        outPath += ".tflite"
        print(f"Adding missing .tflite extension to output path: {outPath}")

    with open(outPath, 'wb') as f:
        f.write(tfliteModel)

    # Loading converted model and printing model details for inspection.
    ip = tf.lite.Interpreter(outPath)
    print(f"- input details for {os.path.basename(outPath)}: {pformat(ip.get_input_details())}")
    print(f"- output details for {os.path.basename(outPath)}: {pformat(ip.get_output_details())}")
    print("Conversion success")

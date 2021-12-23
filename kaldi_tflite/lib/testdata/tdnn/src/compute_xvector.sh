#!/usr/bin/env bash

# This script generates outputs using dummy input given a kaldi nnet3 model model.
# The output can be taken from a specified layer within the model instead of the
# default output layer (e.g. x-vectors extracted from penultimate layer).

# options:
# - e: exit immediately when a pipeline fails
# - u: treat unset variables as an error
# - o pipefail: return value of rightmost command in a pipeline
set -euo pipefail

# Get the top level directory.
TOP=$(git rev-parse --show-toplevel)

# Kaldi repo directory with built binaries.
kaldi="/home/$USER/git/kaldi"

# Model config to generate the model files and referene inputs/outputs for.
model_name="0008_sitw_v2_1a"
# model_name="0006_callhome_diarization_v2_1a"

model_path="${TOP}/data/kaldi_models/${model_name}/exp/xvector_nnet_1a/final.raw"

# Layer to extract the output from.
# output_layer="stats-pooling-0-10000"
output_layer="tdnn6.affine"
# output_layer="tdnn7.affine"

# Output directory where reference input and outputs will be written to.
out_dir="${TOP}/kaldi_tflite/lib/testdata/tdnn/src/${model_name}_${output_layer}"

# Nnet3 model config edits; selects which layer to get the output from.
extract_cfg="output-node name=output input=${output_layer}"

# Input features archive (text format). If provided, will use this instead of
# generating dummy input.
input_feats="${TOP}/kaldi_tflite/lib/testdata/mfcc_chunk_30_16khz.ark.txt"
# input_feats=""

# Model's input feature dimension for generating dummy input. Not used if input
# feats specfied above.
input_dim=23

# Number of frames of dummy input to provide for generating reference output
# from the models. Not used if input feats specified above.
input_num_frames=150

# ---------------------------------

mkdir -p ${out_dir}

# Creating dummy input for the model.
if [[ "${input_feats}" == "" ]]; then
    rm -f "${out_dir}/feat.ark.txt"
    echo "seg1 [" > "${out_dir}/feat.ark.txt"
    for i in $(seq 1 ${input_num_frames}); do
        for j in $(seq 1 ${input_dim}); do
            echo -n "0.$(( i*j )) " >> "${out_dir}/feat.ark.txt"
        done
        echo "" >> "${out_dir}/feat.ark.txt"
    done
    echo "]" >> "${out_dir}/feat.ark.txt"
else
    # Symlinking input features to test data directory. 
    feat_path="${out_dir}/feat.ark.txt"
    ln -sf "$(realpath ${input_feats} --relative-to=${out_dir})" "${feat_path}"
fi

echo "${extract_cfg}" > "${out_dir}/exctract.cfg"

# Computing output from specified layer.
"${kaldi}/src/nnet3bin/nnet3-xvector-compute" \
    --min-chunk-size=${input_num_frames} \
    "${kaldi}/src/nnet3bin/nnet3-copy --nnet-config=\"${out_dir}/exctract.cfg\" \"${model_path}\" - |" \
    ark,t:"${out_dir}/feat.ark.txt" \
    ark,t:"${out_dir}/output.ark.txt"

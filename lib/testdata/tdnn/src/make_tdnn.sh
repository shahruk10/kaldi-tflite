#!/usr/bin/env bash

# This script generates TDNN model files that can be used for testing
# purposes. Along with the models, test input data and the outputs from
# the models using the inputs are also generated.

# options:
# - e: exit immediately when a pipeline fails
# - u: treat unset variables as an error
# - o pipefail: return value of rightmost command in a pipeline
set -euo

# Get the top level directory.
TOP=$(git rev-parse --show-toplevel)

# Kaldi repo directory with built binaries.
kaldi=/home/$USER/git/kaldi

# Model config to generate the model files and referene inputs/outputs for.
model_config="tdnn_narrow"

# Number of frames of input to provide for generating reference output from
# the models.
input_num_frames=8

# Output directory where initialized dummy model along with reference
# inputs/outputs will be written to.
nnet_dir="${TOP}/lib/testdata/tdnn/src/${model_config}"

# If true, will generate new model files.
gen_model_files=false

# If true, will generate inputs and outputs from the model.
gen_inputs_and_outputs=true


function tdnn_single_layer() {
    out_dir=$1
    mkdir -p ${out_dir}/configs

    feat_dim=30
    layer_units=32
    layer_context="(-3, -1, 0, 1)"

    cat <<EOF > ${out_dir}/configs/network.xconfig
    # please note that it is important to have input layer with the name=input

    # The frame-level layers
    input dim=${feat_dim} name=input
    relu-layer name=tdnn1 input=Append${layer_context} dim=${layer_units}
    output name=output
EOF
}

function tdnn_narrow() {
    out_dir=$1
    mkdir -p ${out_dir}/configs

    feat_dim=3

    cat <<EOF > ${out_dir}/configs/network.xconfig
    # please note that it is important to have input layer with the name=input

    # The frame-level layers
    input dim=${feat_dim} name=input
    relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=5
    relu-batchnorm-layer name=tdnn2 input=Append(-2,0,2) dim=8
    relu-batchnorm-layer name=tdnn3 input=Append(-3,0,3) dim=8
    relu-batchnorm-layer name=tdnn4 dim=8
    relu-batchnorm-layer name=tdnn5 dim=8
    output-layer name=output dim=1
EOF
}

function tdnn_xvector() {
    out_dir=$1
    mkdir -p ${out_dir}/configs

    feat_dim=30
    num_targets=100

    # This chunk-size corresponds to the maximum number of frames the
    # stats layer is able to pool over.  In this script, it corresponds
    # to 100 seconds.  If the input recording is greater than 100 seconds,
    # we will compute multiple xvectors from the same recording and average
    # to produce the final xvector.
    max_chunk_size=10000

    # The smallest number of frames we're comfortable computing an xvector from.
    # Note that the hard minimum is given by the left and right context of the
    # frame-level layers.
    min_chunk_size=25

    cat <<EOF > ${out_dir}/configs/network.xconfig
    # please note that it is important to have input layer with the name=input

    # The frame-level layers
    input dim=${feat_dim} name=input
    relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=512
    relu-batchnorm-layer name=tdnn2 input=Append(-2,0,2) dim=512
    relu-batchnorm-layer name=tdnn3 input=Append(-3,0,3) dim=512
    relu-batchnorm-layer name=tdnn4 dim=512
    relu-batchnorm-layer name=tdnn5 dim=1500

    # The stats pooling layer. Layers after this are segment-level.
    # In the config below, the first and last argument (0, and ${max_chunk_size})
    # means that we pool over an input segment starting at frame 0
    # and ending at frame ${max_chunk_size} or earlier.  The other arguments (1:1)
    # mean that no subsampling is performed.
    stats-layer name=stats config=mean+stddev(0:1:1:${max_chunk_size})

    # This is where we usually extract the embedding (aka xvector) from.
    relu-batchnorm-layer name=tdnn6 dim=512 input=stats

    # This is where another layer the embedding could be extracted
    # from, but usually the previous one works better.
    relu-batchnorm-layer name=tdnn7 dim=512
    output-layer name=output include-log-softmax=true dim=${num_targets}
EOF
}

# Input dimensions of the models defined above.
declare -A model_input_dims=(
    ["tdnn_single_layer"]=30
    ["tdnn_narrow"]=3
    ["tdnn_xvector"]=30
)

if [[ "${gen_model_files}" == true ]]; then

    if [ ${model_config} == "tdnn_single_layer" ]; then
        tdnn_single_layer ${nnet_dir}
    elif [ ${model_config} == "tdnn_narrow" ]; then
        tdnn_narrow ${nnet_dir}
    elif [ ${model_config} == "tdnn_xvector" ]; then
        tdnn_xvector ${nnet_dir}
    else
        echo "ERROR: unknown architecture type: ${model_config}"
        exit 1
    fi

    mkdir -p ${nnet_dir}

    # Converting config into model definition files parsable by Kaldi binaries.
    PATH="${kaldi}/src/nnet3bin:${PATH}" PYTHONPATH=${kaldi}/egs/wsj/s5/steps \
        ${kaldi}/egs/wsj/s5/steps/nnet3/xconfig_to_configs.py \
            --xconfig-file ${nnet_dir}/configs/network.xconfig \
            --config-dir ${nnet_dir}/configs/

    # Randomly initializing paramters of the defined model.
    ${kaldi}/src/nnet3bin/nnet3-init \
        --srand=-2 \
        --binary=true \
        ${nnet_dir}/configs/final.config \
        ${nnet_dir}/final.raw

    # Saving model as a readable text file as well for debugging.
    ${kaldi}/src/nnet3bin/nnet3-copy \
        --binary=false \
        ${nnet_dir}/final.raw \
        ${nnet_dir}/final.raw.txt
fi

if [[ "${gen_inputs_and_outputs}" == true ]]; then

    # Creating dummy input for the model.
    input_dim="${model_input_dims[${model_config}]}"

    echo "seg1 [" > ${nnet_dir}/feat.ark.txt
    for i in $(seq 1 ${input_num_frames}); do
        for j in $(seq 1 ${input_dim}); do
            echo -n "0.$(( i*j )) " >> ${nnet_dir}/feat.ark.txt
        done
        echo "" >> ${nnet_dir}/feat.ark.txt
    done
    echo "]" >> ${nnet_dir}/feat.ark.txt

    # Computing output using the dummy input data.
    ${kaldi}/src/nnet3bin/nnet3-compute \
        ${nnet_dir}/final.raw.txt \
        ark,t:${nnet_dir}/feat.ark.txt \
        ark,t:${nnet_dir}/output.ark.txt
fi

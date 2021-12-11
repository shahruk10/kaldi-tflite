#!/usr/bin/env bash

# This script generates Kaldi nnet3 model files containing StatisticsExtraction
# and StatisticsPooling components that can be used for testing purposes. Along
# with the models, test input data and the outputs from the models using the
# inputs are also generated.

# options:
# - e: exit immediately when a pipeline fails
# - u: treat unset variables as an error
# - o pipefail: return value of rightmost command in a pipeline
set -euo pipefail

# Get the top level directory.
TOP=$(git rev-parse --show-toplevel)

# Kaldi repo directory with built binaries.
kaldi=/home/$USER/git/kaldi

# Model config to generate the model files and referene inputs/outputs for.
# model_config="stats_mean"
# model_config="stats_mean_std"
# model_config="stats_mean_std_windowed"
# model_config="stats_mean_std_only_left_context"
# model_config="stats_mean_std_both_left_right_context"
# model_config="stats_mean_std_asymmetrical_context"
# model_config="stats_mean_std_subsampling"
model_config="stats_mean_std_windowed_subsampling"


# Number of frames of input to provide for generating reference output from
# the models.
input_num_frames=16

# Length of input feature dimension.
input_dim=3

# Output directory where initialized dummy model along with reference
# inputs/outputs will be written to.
nnet_dir="${TOP}/lib/testdata/stats/src/${model_config}"

# If true, will generate new model files.
gen_model_files=true

# If true, will generate inputs and outputs from the model.
gen_inputs_and_outputs=true


function stats_pooling_mdl() {
    out_dir=$1
    stats=$2
    left_context=$3
    input_period=$4
    output_period=$5
    right_context=$6

    mkdir -p ${out_dir}/configs

    sampling_config="${left_context}:${input_period}:${output_period}:${right_context}"

    cat <<EOF > ${out_dir}/configs/network.xconfig

    input dim=${input_dim} name=input
    stats-layer name=stats config=${stats}(${sampling_config})
    output name=output
EOF
}


if [[ "${gen_model_files}" == true ]]; then

    case "${model_config}" in 
        "stats_mean")
            stats_pooling_mdl ${nnet_dir} "mean" 0 1 1 ${input_num_frames}
            ;;
        "stats_mean_std")
            stats_pooling_mdl ${nnet_dir} "mean+stddev" 0 1 1 ${input_num_frames}
            ;;
        "stats_mean_std_windowed")
            ctx=$(( input_num_frames / 4))
            stats_pooling_mdl ${nnet_dir} "mean+stddev" 0 1 1 ${ctx}
            ;;
        "stats_mean_std_only_left_context")
            ctx=$(( input_num_frames / 4))
            stats_pooling_mdl ${nnet_dir} "mean+stddev" -${ctx} 1 1 0
            ;;
       "stats_mean_std_both_left_right_context")
            ctx=$(( input_num_frames / 4))
            stats_pooling_mdl ${nnet_dir} "mean+stddev" -${ctx} 1 1 ${ctx}
            ;;
        "stats_mean_std_asymmetrical_context")
            lctx=$(( input_num_frames / 4))
            rctx=$(( input_num_frames / 8))
            stats_pooling_mdl ${nnet_dir} "mean+stddev" -${lctx} 1 1 ${rctx}
            ;;
        "stats_mean_std_subsampling")
            stats_pooling_mdl ${nnet_dir} "mean+stddev" 0 4 4 ${input_num_frames}
            ;;
        "stats_mean_std_windowed_subsampling")
            ctx=$(( input_num_frames / 4))
            stats_pooling_mdl ${nnet_dir} "mean+stddev" -${ctx} 4 4 ${ctx}
            ;;

        *)
            echo "ERROR: unknown config type: ${model_config}"
            exit 1
    esac

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
    echo "seg1 [" > ${nnet_dir}/feat.ark.txt
    for i in $(seq 1 ${input_num_frames}); do
        for j in $(seq 1 ${input_dim}); do
            echo -n "$(( i*j )) " >> ${nnet_dir}/feat.ark.txt
        done
        echo "" >> ${nnet_dir}/feat.ark.txt
    done
    echo "]" >> ${nnet_dir}/feat.ark.txt

    # Computing output using the dummy input data.
    ${kaldi}/src/nnet3bin/nnet3-compute \
        ${nnet_dir}/final.raw \
        ark,t:${nnet_dir}/feat.ark.txt \
        ark,t:${nnet_dir}/output.ark.txt
fi

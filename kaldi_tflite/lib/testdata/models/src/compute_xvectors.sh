#!/usr/bin/env bash

# This script computes audio features using kaldi on input audio files. The
# generated features can be used as reference data or input to other test cases.

# options:
# - e: exit immediately when a pipeline fails
# - u: treat unset variables as an error
# - o pipefail: return value of rightmost command in a pipeline
set -euo pipefail

# Get the top level directory.
TOP=$(git rev-parse --show-toplevel)

# Kaldi repo directory with built binaries.
kaldi="/home/$USER/git/kaldi"

# ---------------------------------

function write_conf_0008_sitw_v2_1a() {
    out_dir=$1

    cat <<EOF > ${out_dir}/mfcc.conf
      --sample-frequency=16000
      --frame-length=25
      --low-freq=20
      --high-freq=7600
      --num-mel-bins=30
      --num-ceps=30
      --snip-edges=false        
EOF

    cat <<EOF > ${out_dir}/cmvn.conf
      --norm-vars=false
      --center=true
      --cmn-window=300      
EOF

    cat <<EOF > ${out_dir}/vad.conf
      --vad-energy-threshold=5.5
      --vad-energy-mean-scale=0.5
      --vad-proportion-threshold=0.12
      --vad-frames-context=2      
EOF
}

function compute_mfcc() {
    out_dir=$1
    input_scp=$2

    # Computing MFCCs.
    "${kaldi}/src/featbin/compute-mfcc-feats" \
        --config="${out_dir}/mfcc.conf" \
        scp,p:<(echo "${input_scp}") \
        ark,scp,t:"${out_dir}/feats.ark.txt,${out_dir}/feats.scp"
}

function compute_vad() {
    out_dir=$1

    # Computing VAD.
    "${kaldi}/src/ivectorbin/compute-vad" \
        --config="${out_dir}/vad.conf" \
        scp:"${out_dir}/feats.scp" \
        ark,scp,t:"${out_dir}/vad.ark.txt,${out_dir}/vad.scp"
}

function apply_cmvn() {
    out_dir=$1
    mfcc_dir=$2

    ln -sf "$(realpath ${mfcc_dir}/mfcc.ark.txt --relative-to=${out_dir})" "${out_dir}/mfcc.ark.txt"

    # Applying CMVN.
    "${kaldi}/src/featbin/apply-cmvn-sliding" \
        --config="${out_dir}/cmvn.conf" \
        ark,t:"${out_dir}/mfcc.ark.txt" \
        ark,t:"${out_dir}/cmvn.ark.txt"
}

models_dir="${TOP}/data/kaldi_models"
audio_path="${TOP}/kaldi_tflite/lib/testdata/librispeech_2.wav"
input_scp="seg1 ${audio_path}"
spk2utt="seg1 seg1"

models=( 0008_sitw_v2_1a )

# The steps followed below to extract x-vectors follows those in the kaldi
# script: sid/nnet3/xvector/extract_xvectors.sh 
for model_name in "${models[@]}"; do

    nnet_dir="${models_dir}/${model_name}/exp/xvector_nnet_1a"

    out_dir="${TOP}/kaldi_tflite/lib/testdata/models/src/${model_name}"
    rm -rf "${out_dir}" && mkdir -p "${out_dir}"

    # Writing spk2utt file for the extract_xvectors.sh script.
    echo "${spk2utt}" > "${out_dir}/spk2utt"

    # Soft linking test audio file to output directory for reference.
    ln -sf "$(realpath ${audio_path} --relative-to=${out_dir})" "${out_dir}/audio.wav"

    # Writing feat confs to output directory.
    $"write_conf_${model_name}" "${out_dir}"

    # Extracting MFCCs and computing VAD.
    compute_mfcc "${out_dir}" "${input_scp}"
    compute_vad "${out_dir}"

    # Working around hard-coded data split in the extract_xvectors.sh script.
    mkdir -p "${out_dir}/split1/1"
    cp "${out_dir}/"{vad.scp,feats.scp} "${out_dir}/split1/1/"
 
    # Extracting x-vectors.
    pushd "${kaldi}/egs/sre08/v1"
    "${kaldi}/egs/sre08/v1/sid/nnet3/xvector/extract_xvectors.sh" \
      --nj 1 \
      "${nnet_dir}" \
      "${out_dir}" \
      "${out_dir}"
    popd

    "${kaldi}/src/bin/copy-vector" scp:"${out_dir}/xvector.scp" ark,t:"${out_dir}/xvector.unnorm.ark.txt"

    # Subtracting global mean, applying LDA transform and length normalization.
    "${kaldi}/src/ivectorbin/ivector-subtract-global-mean" \
        "${nnet_dir}/xvectors_train_combined_200k/mean.vec" \
        scp:"${out_dir}/xvector.scp" \
        ark:- | \
        "${kaldi}/src/bin/transform-vec" \
            "${nnet_dir}/xvectors_train_combined_200k/transform.mat" \
            ark:- \
            ark:- | \
                "${kaldi}/src/ivectorbin/ivector-normalize-length" ark:- ark,t:"${out_dir}/xvector.ark.txt"

    # Cleaning up intermediate files.
    keep=(
        "${out_dir}/xvector.ark.txt"
        "${out_dir}/xvector.unnorm.ark.txt"
        "${out_dir}/audio.wav"
        "${out_dir}/mfcc.conf"
        "${out_dir}/vad.conf"
        "${out_dir}/cmvn.conf"
    )

    for f in "${out_dir}"/*; do
        skip=false
        for kf in "${keep[@]}"; do
            if  [[ "${f}" = "${kf}" ]]; then
                skip=true
                break
            fi
        done

        if [[ "${skip}" == true ]]; then
            continue
        fi

        rm -rf "${f}"
    done
done

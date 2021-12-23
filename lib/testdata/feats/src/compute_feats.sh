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

function write_feat_conf() {
    out_dir=$1
    sample_freq=$2
    num_mels=$3
    snip_edges=$4
    frame_length=$5
    frame_shift=$6
    raw_energy=$7
    num_ceps=$8

    cat <<EOF > ${out_dir}/mfcc.conf
        --sample-frequency=${sample_freq}
        --frame-length=${frame_length}
        --frame-shift=${frame_shift}
        --raw-energy=${raw_energy}
        --dither=0
        --low-freq=20
        --high-freq=-400
        --num-mel-bins=${num_mels}
        --num-ceps=${num_ceps}
        --snip-edges=${snip_edges}
EOF

    cat <<EOF > ${out_dir}/fbank.conf
        --sample-frequency=${sample_freq}
        --frame-length=${frame_length}
        --frame-shift=${frame_shift}
        --raw-energy=${raw_energy}
        --dither=0
        --low-freq=20
        --high-freq=-400
        --num-mel-bins=${num_mels}
        --use-log-fbank=true
        --use-power=true
        --snip-edges=${snip_edges}
EOF
}

function generate_feats() {
    out_dir=$1
    audio_path=$2
    input_scp=$3

    ln -sf "$(realpath ${audio_path} --relative-to=${out_dir})" "${out_dir}/audio.wav"

    # Computing Filter bank features.
    "${kaldi}/src/featbin/compute-fbank-feats" \
        --config="${out_dir}/fbank.conf" \
        scp,p:<(echo "${input_scp}") \
        ark,t:"${out_dir}/fbank.ark.txt"

    # Computing MFCCs.
    "${kaldi}/src/featbin/compute-mfcc-feats" \
        --config="${out_dir}/mfcc.conf" \
        scp,p:<(echo "${input_scp}") \
        ark,t:"${out_dir}/mfcc.ark.txt"
}

function write_cmvn_conf() {
    out_dir=$1
    window=$2
    norm_vars=$3
    center=$4
    min_window=$5

    cat <<EOF > ${out_dir}/cmvn.conf
        --cmn-window=${window}
        --norm-vars=${norm_vars}
        --center=${center}
        --min-cmn-window=${min_window}
EOF
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

for sample_freq in 16000; do
    n=0
    for num_mels in 23 30; do
        for snip_edges in true false; do
            for frame_length in 25 32; do
                for frame_shift in 10 16; do
                    for raw_energy in true false; do
                        for num_ceps in 23 30; do
                            if [[ ${num_ceps} -gt ${num_mels} ]]; then
                                continue
                            fi

                            n=$((n+1))
                            audio_path="${TOP}/lib/testdata/librispeech_2_trimmed.wav"
                            input_scp="seg1 ${audio_path}"

                            # Output directory where reference input and outputs will be written to.
                            id=$(printf "%03d" ${n})
                            out_dir="${TOP}/lib/testdata/feats/src/fbank_mfcc/${sample_freq}_${id}"
                            mkdir -p "${out_dir}"

                            write_feat_conf "${out_dir}" ${sample_freq} ${num_mels} ${snip_edges} ${frame_length} ${frame_shift} ${raw_energy} ${num_ceps}
                            generate_feats "${out_dir}" "${audio_path}" "${input_scp}"
                        done
                    done
                done
            done
        done
    done
done

# Applying CMVN with various conifgs on a single set of MFCCs.
mfcc_dir="${TOP}/lib/testdata/feats/src/fbank_mfcc/16000_001"
n=0
for center in true; do
    for min_window in 100; do
        for window in 199 200 201 600; do
            for norm_vars in false true; do
                n=$((n+1))
                id=$(printf "%03d" ${n})
                out_dir="${TOP}/lib/testdata/feats/src/cmvn/16000_001_${id}"
                mkdir -p "${out_dir}"

                write_cmvn_conf "${out_dir}" ${window} ${norm_vars} ${center} ${min_window}
                apply_cmvn "${out_dir}" "${mfcc_dir}"
            done
        done
    done
done

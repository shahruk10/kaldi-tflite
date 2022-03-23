#!/usr/bin/env bash

# This script generates dummy ivector models. It does so first in text form, and
# then uses kaldi's `ivector-extractor-copy` binary to convert it to the binary
# form. This also validates that the dummy i-vector model is in the valid
# format. The generated models can be used as reference data or used in other
# tests.

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

function generate_ivec_extractor() {
    out_dir=$1
    num_gauss=$2
    feat_dim=$3
    ivec_dim=$4

    # Constructing mixture weights.
    wvec="["
    for i in $(seq 1 ${num_gauss}); do
        wvec="${wvec} $i.0"
    done
    wvec="${wvec} ]"

    # Constructing ivector projection matrix of dimension (feat_dim x ivec_dim).
    # Will reuse the same matrix for each gaussian.
    M="["
    for i in $(seq 1 ${feat_dim}); do
        M="${M}\n "
        for j in $(seq 1 ${ivec_dim}); do
            M="${M} $(( i*j )).0"
        done
    done
    M="${M} ]"

    # Repeat the projection matrix created above, and trim last newline.
    proj_mats=""
    for _ in $(seq 1 ${num_gauss}); do
        proj_mats="${proj_mats}${M}\n"
    done
    proj_mats="${proj_mats%\\n}"

    # Constructing matrix containing inverse variances of speaker-adapted model.
    # This matrix has a dimension of (feat_dim x feat_dim). It also must be
    # symmetric and postive definite. Inside the model file, only the lower
    # triangular portion of the matrix is stored. For simplicity we construct a
    # diagonal matrix which satiesfies all criteria.
    #
    # Will reuse the same matrix for each gaussian.
    sigma_inv="["
    for i in $(seq 1 ${feat_dim}); do
        sigma_inv="${sigma_inv}\n "
        for j in $(seq 1 ${i}); do
            if [[ ${j} -eq ${i} ]]; then
                sigma_inv="${sigma_inv} ${i}.0"
            else
                sigma_inv="${sigma_inv} 0.0"
            fi
        done
    done
    sigma_inv="${sigma_inv} ]"

    # Repeat the inverse variance matrix created above, and trim last newline.
    sigma_inv_mats=""
    for _ in $(seq 1 ${num_gauss}); do
        sigma_inv_mats="${sigma_inv_mats}${sigma_inv}\n"
    done
    sigma_inv_mats="${sigma_inv_mats%\\n}"

    # Constructing prior offset. Using sum of test params.
    prior_offset="$(( num_gauss + feat_dim + ivec_dim )).0"

    # Writing out the text form.
    { echo -e "<IvectorExtractor> <w>  [ ]";
      echo -e "<w_vec>  ${wvec}";
      echo -e "<M> ${num_gauss} ${proj_mats}";
      echo -e "<SigmaInv> ${sigma_inv_mats}";
      echo -e "<IvectorOffset> ${prior_offset} </IvectorExtractor>"; } > "${out_dir}/final.ie.txt"

    # Converting the text form to binary.
    "${kaldi}/src/ivectorbin/ivector-extractor-copy" \
        --binary=true "${out_dir}/final.ie.txt" "${out_dir}/final.ie"

    # Write out the test params generated matrices use to generate the model so
    # that it can be loaded by test scripts easily.
    echo -e "${M}" > "${out_dir}/M.mat.txt"
    echo -e "${sigma_inv}" > "${out_dir}/sigma_inv.mat.txt"
    {  echo "numGauss=${num_gauss}";
       echo "featDim=${feat_dim}";
       echo "ivecDim=${ivec_dim}";
       echo "priorOffset=${prior_offset}"; } > "${out_dir}/test_params.txt"
}

n=0
for num_gauss in 2 4 6; do
    for feat_dim in 2 4 6; do
        for ivec_dim in 4 8; do 
            if [[ ${feat_dim} -gt ${ivec_dim} ]]; then
                continue
            fi

            n=$((n+1))
            id=$(printf "%03d" ${n})
            out_dir="${TOP}/kaldi_tflite/lib/testdata/ivector_extractor/src/dummy_ie_models/dummy_${id}"
            mkdir -p "${out_dir}"

            generate_ivec_extractor "${out_dir}" ${num_gauss} ${feat_dim} ${ivec_dim}
        done
    done
done

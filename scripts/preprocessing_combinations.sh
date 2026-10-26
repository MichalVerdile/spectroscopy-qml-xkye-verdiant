#!/usr/bin/env bash
# Sourced by the 600-point runners (CNN, MPS, TTN, XGBoost).
#
# Sweeps every non-empty combination of the four preprocessing steps
#   resample=fir, smooth=gaussian:2, repr=sqrt, norm=snv
# (2^4 - 1 = 15 combinations; the empty combination is the plain linear baseline
# `shared_mssd`, run it with PREPROCESSING=shared_mssd). Each combination gets its
# own experiment directory because one directory must not mix preprocessing specs.

# Prints "<name><TAB><spec>" for each of the 15 combinations.
list_preprocessing_combinations() {
    local mask bit name spec
    local -a steps=("fir" "gaussian" "sqrt" "snv")
    local -a keys=("resample=fir" "smooth=gaussian:2" "repr=sqrt" "norm=snv")
    for ((mask = 1; mask < 16; mask++)); do
        name=""
        spec=""
        for bit in 0 1 2 3; do
            if ((mask & (1 << bit))); then
                name+="${name:+_}${steps[bit]}"
                spec+="${spec:+,}${keys[bit]}"
            fi
        done
        printf '%s\t%s\n' "$name" "$spec"
    done
}

# Usage: if sweep_preprocessing_combinations "$0" "$ROOT_DIR"; then exit 0; fi
# With PREPROCESSING unset, re-runs the calling script once per combination
# (EXPERIMENT_DIR=<root>/<name>) and returns 0. With PREPROCESSING set, returns 1
# so the caller runs that single preprocessing as before.
sweep_preprocessing_combinations() {
    local script="$1" root="$2" name spec
    [[ -z "${PREPROCESSING:-}" ]] || return 1

    while IFS=$'\t' read -r name spec; do
        echo "##### Preprocessing combination: $name ($spec) #####"
        PREPROCESSING="$spec" EXPERIMENT_DIR="$root/$name" bash "$script"
    done < <(list_preprocessing_combinations)
    return 0
}

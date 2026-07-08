#!/bin/bash

run_resp_stage_1() {
    local pref=$1
    local dat_pref=$2
    resp -O -i "${pref}.in" -o "${pref}_${dat_pref}.out" -t "${pref}_${dat_pref}.chg" -s "${pref}_${dat_pref}.esp" -e "${dat_pref}_gaussian.dat"
}

# run_resp "stage_1" "acetic_acid_a"
# run_resp "stage_1_x" "acetic_acid_a_x"

# run_resp "stage_1" "acetic_acid_b"
# run_resp "stage_1_x" "acetic_acid_b_x"

# run_resp "stage_1" "acetic_acid_c"
# run_resp "stage_1_x" "acetic_acid_c_x"

# run_resp "stage_1" "acetic_acid_d"
# run_resp "stage_1_x" "acetic_acid_d_x"

# cat acetic_acid_a_gaussian.dat acetic_acid_b_gaussian.dat acetic_acid_c_gaussian.dat acetic_acid_d_gaussian.dat > acetic_acid_gaussian.dat
# cat acetic_acid_a_x_gaussian.dat acetic_acid_b_x_gaussian.dat acetic_acid_c_x_gaussian.dat acetic_acid_d_x_gaussian.dat > acetic_acid_x_gaussian.dat

run_resp_stage_1 "stage_1" "acetic_acid"
run_resp_stage_1 "stage_1_x" "acetic_acid_x"

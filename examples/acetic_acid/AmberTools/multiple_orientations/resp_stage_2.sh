#!/bin/bash


run_resp_stage_2() {
    local pref=$1
    local dat_pref=$2
    local charges_source=$3
    resp -O -i "${pref}.in" -o "${pref}_${dat_pref}.out" -t "${pref}_${dat_pref}.chg" -s "${pref}_${dat_pref}.esp" -e "${dat_pref}_gaussian.dat" -q ""${charges_source}_${dat_pref}.chg""
}

run_resp_stage_2 "stage_2" "acetic_acid" "stage_1"
run_resp_stage_2 "stage_2_x" "acetic_acid_x" "stage_1_x"

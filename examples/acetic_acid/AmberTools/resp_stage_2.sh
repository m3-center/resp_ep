#!/bin/bash -f

resp -O \
    -i stage_2.in \
    -o stage_2.out \
    -q stage_1.chg \
    -t stage_2.chg \
    -s stage_2.esp \
    -e acetic_acid_gaussian.dat

resp -O \
    -i stage_2_x.in \
    -o stage_2_x.out \
    -q stage_1_x.chg \
    -t stage_2_x.chg \
    -s stage_2_x.esp \
    -e acetic_acid_x_gaussian.dat

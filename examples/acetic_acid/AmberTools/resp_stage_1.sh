#!/bin/bash -f

resp -O \
    -i stage_1.in \
    -o stage_1.out \
    -t stage_1.chg \
    -s stage_1.esp \
    -e acetic_acid_gaussian.dat

resp -O \
    -i stage_1_x.in \
    -o stage_1_x.out \
    -t stage_1_x.chg \
    -s stage_1_x.esp \
    -e acetic_acid_x_gaussian.dat


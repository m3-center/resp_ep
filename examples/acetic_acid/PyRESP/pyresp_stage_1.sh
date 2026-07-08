#!/bin/bash -f

echo " "
echo "py_resp stage 1"
echo " "

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYRESP_DIR="${SCRIPT_DIR}/../../../../PyRESP"

# Check if py_resp.py exists
if [ ! -f "${PYRESP_DIR}/py_resp.py" ]; then
    echo "Error: py_resp.py not found at ${PYRESP_DIR}/py_resp.py"
    exit 1
fi

"${PYRESP_DIR}/py_resp.py" -O \
		-i stage_1.in \
		-o stage_1.out \
		-t stage_1.chg \
		-s stage_1.esp \
		-e acetic_acid_gaussian.dat

"${PYRESP_DIR}/py_resp.py" -O \
		-i stage_1_x.in \
		-o stage_1_x.out \
		-t stage_1_x.chg \
		-s stage_1_x.esp \
		-e acetic_acid_x_gaussian.dat

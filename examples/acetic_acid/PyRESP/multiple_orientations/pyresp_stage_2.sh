#!/bin/bash -f

echo " "
echo "py_resp demo: py_resp on water"
echo " "

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYRESP_DIR="${SCRIPT_DIR}/../../../../../PyRESP"

# Check if py_resp.py exists
if [ ! -f "${PYRESP_DIR}/py_resp.py" ]; then
    echo "Error: py_resp.py not found at ${PYRESP_DIR}/py_resp.py"
    exit 1
fi

"${PYRESP_DIR}/py_resp.py" -O \
		-i stage_2.in \
		-o stage_2.out \
		-q stage_1.chg \
		-t stage_2.chg \
		-s stage_2.esp \
		-e acetic_acid_gaussian.dat

"${PYRESP_DIR}/py_resp.py" -O \
		-i stage_2_x.in \
		-o stage_2_x.out \
		-q stage_1_x.chg \
		-t stage_2_x.chg \
		-s stage_2_x.esp \
		-e acetic_acid_x_gaussian.dat
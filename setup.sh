#!/bin/bash
# setup mlir-aie environment
export AIETOOLS_ROOT=/tools/ryzen_ai-1.3.0/vitis_aie_essentials
export PATH=$PATH:${AIETOOLS_ROOT}/bin
export LM_LICENSE_FILE=/tools/Xilinx.lic

MLIR_AIE_BUILD_DIR=~/mlir-aie

source /opt/xilinx/xrt/setup.sh

source ${MLIR_AIE_BUILD_DIR}/ironenv/bin/activate
source ${MLIR_AIE_BUILD_DIR}/utils/env_setup.sh


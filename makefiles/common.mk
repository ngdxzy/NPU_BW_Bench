# common.mk
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Created by Alfred

ifneq ($(filter -j,$(MAKEFLAGS)),)
    # Nothing to do; -j is already set
else
    MAKEFLAGS += -j8
endif


# Kernel stuff
WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined
AIEOPT_DIR := $(shell realpath $(dir $(shell which aie-opt))/..)
LLVM_AIE_DIR := $(shell realpath $(dir $(shell which aie-opt))/../../)
ifeq ($(DEVICE), npu2) # Strix point
	AIETOOLS_DIR := $(shell realpath $(dir $(shell which xchesscc))/../..)
	AIE_INCLUDE_DIR := ${AIETOOLS_DIR}/data/versal_prod/lib
	AIE2_INCLUDE_DIR := ${AIETOOLS_DIR}/data/aie_ml/lib
else # Hawkeye point
	VITIS_ROOT := $(shell realpath $(dir $(shell which xchesscc))/../..)
	AIETOOLS_DIR := ${VITIS_ROOT}/aietools
	AIE_INCLUDE_DIR := ${VITIS_ROOT}/aietools/data/versal_prod/lib
	AIE2_INCLUDE_DIR := ${VITIS_ROOT}/aietools/data/aie_ml/lib

	CHESS_FLAGS = -P ${AIE_INCLUDE_DIR}
	PEANO_INSTALL_DIR := ${LLVM_AIE_DIR}/llvm-aie
	PEANOWRAP2_FLAGS = -O2 -v -std=c++20 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include -I /tools/Xilinx/Vitis/2023.2/aietools/include
endif

CHESSCCWRAP1_FLAGS = aie -I ${AIETOOLS_DIR}/include 
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include
CHESSCCWRAP2P_FLAGS = aie2p -I ${AIETOOLS_DIR}/include 

CHESSCC1_FLAGS = -f -p me -P ${AIE_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
CHESSCC2_FLAGS = -f -p me -P ${AIE2_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include -D__AIENGINE__=2 -D__AIEARCH__=20
CHESS_FLAGS = -P ${AIE_INCLUDE_DIR}

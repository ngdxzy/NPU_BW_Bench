#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified by Alfred

include makefiles/common.mk

DEVICE ?= npu2
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
# Bitstream makefile
include makefiles/bitstream.mk
include makefiles/mlir_bitstream.mk


# Host makefile
include makefiles/host.mk

.PHONY: run all kernel link bitstream host clean instructions
all: ${XCLBIN_TARGET} ${INSTS_TARGET} ${HOST_C_TARGET}

clean:
	-@rm -rf build 
	-@rm -rf log
	-@rm -rf host.exe
	-@rm -rf trace*

test:
	echo "test"
	echo ${AIEOPT_DIR}


instructions: ${INSTS_TARGETS}


link: ${MLIR_TARGET} 


bitstream: ${XCLBIN_TARGETS}


host: ${HOST_C_TARGET}


clean_host:
	-@rm -rf build/host


run: ${HOST_C_TARGET} ${XCLBIN_TARGET} ${INSTS_TARGET}
	./${HOST_C_TARGET} | tee out.log

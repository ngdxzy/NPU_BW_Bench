# common.mk
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Created by Alfred

# Kernel stuff
KERNEL_O_DIR := build/kernel
KERNEL_SRCS := $(wildcard ${HOME_DIR}/kernel/*.cc)
KERNEL_OBJS := $(patsubst ${HOME_DIR}/kernel/%.cc, ${KERNEL_O_DIR}/%.o, $(KERNEL_SRCS))
KERNEL_HEADERS := $(wildcard ${HOME_DIR}/kernel/*.h)

# cd build && /home/alfred/mlir-aie/ironenv/lib/python3.12/site-packages/llvm-aie/bin/clang++ -O2 -v -std=c++20 --target=aie2p-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined -DNDEBUG -I /home/alfred/mlir-aie/ironenv/lib/python3.12/site-packages/mlir_aie/include  -DBIT_WIDTH=8 -c /home/alfred/mlir-aie/programming_examples/basic/passthrough_kernel/../../../aie_kernels/generic/passThrough.cc -o passThrough.cc.o

# cd buildl && /home/alfred/mlir-aie/ironenv/lib/python3.12/site-packages/llvm-aie/bin/clang++ -O2 -v -std=c++20 --target=aie2p-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined -DNDEBUG -I /home/alfred/mlir-aie/ironenv/lib/python3.12/site-packages/mlir_aie/include  -DBIT_WIDTH=8 -c /home/alfred/Projects/Template/kernel/mvm_i8.cc -o mvm_i8.o

# Build kernels
${KERNEL_O_DIR}/%.o: ${HOME_DIR}/kernel/%.cc ${KERNEL_HEADERS}
	mkdir -p ${@D}
ifeq ($(DEVICE),npu1)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else ifeq ($(DEVICE),npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -DBIT_WIDTH=8 -c $< -o ${@F}
else
	echo "Device type not supported"
endif

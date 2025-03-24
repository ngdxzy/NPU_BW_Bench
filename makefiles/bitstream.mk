
# MLIR stuff

MLIR_O_DIR := build/mlir
BITSTREAM_O_DIR := build/bitstream


IRON_BOTH_SRC := $(wildcard ${HOME_DIR}/iron/*.py)
IRON_BOTH_MLIR_TARGET := $(patsubst ${HOME_DIR}/iron/%.py, ${MLIR_O_DIR}/%.mlir, ${IRON_BOTH_SRC})
IRON_BOTH_XCLBIN_TARGET := $(patsubst ${HOME_DIR}/iron/%.py, ${BITSTREAM_O_DIR}/from_iron/%.xclbin, ${IRON_BOTH_SRC})
IRON_BOTH_INSTS_TARGET := $(patsubst ${HOME_DIR}/iron/%.py, ${BITSTREAM_O_DIR}/from_iron/%.txt, ${IRON_BOTH_SRC})

.PRECIOUS: ${IRON_BOTH_MLIR_TARGET} 

# iron generate mlir
${MLIR_O_DIR}/%.mlir: ${HOME_DIR}/iron/%.py
	mkdir -p ${@D}
	python3 $< ${DEVICE} > $@

# Build xclbin
${BITSTREAM_O_DIR}/from_iron/%.xclbin: ${MLIR_O_DIR}/%.mlir ${KERNEL_OBJS}
	mkdir -p ${@D}
	cp ${KERNEL_OBJS} ${@D}
	cd ${@D} && aiecc.py --no-xchesscc --no-xbridge \
		--aie-generate-cdo --aie-generate-xclbin --no-compile-host \
		--xclbin-name=${@F} $(<:${MLIR_O_DIR}/%=../../mlir/%)
	mkdir -p build/xclbins
	cp ${@} build/xclbins/

# $(foreach type,$(INST_TYPES),$(eval $(call AIECC_RULE,$(type))))
${BITSTREAM_O_DIR}/from_iron/%.txt: ${MLIR_O_DIR}/%.mlir
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --no-xchesscc --no-xbridge \
		--aie-generate-npu-insts --no-compile-host -n \
		--npu-insts-name=${@F} $(<:${MLIR_O_DIR}/%=../../mlir/%)
	mkdir -p build/insts
	cp ${@} build/insts/

INSTS_TARGETS += ${IRON_BOTH_INSTS_TARGET}
XCLBIN_TARGETS += ${IRON_BOTH_XCLBIN_TARGET} 

#ifndef __NPU_INSTR_UTILS_HPP__
#define __NPU_INSTR_UTILS_HPP__

#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdio.h>
#include "vector_view.hpp"
#include "debug_utils.hpp"
#include "xrt/xrt_bo.h"
#include "instr_utils/npu_cmd.hpp"
#include "instr_utils/npu_cmd_queue_write.hpp"
#include "instr_utils/npu_cmd_ddr.hpp"
#include "instr_utils/npu_cmd_write_dma.hpp"
#include "instr_utils/npu_cmd_issue_token.hpp"
#include "instr_utils/npu_cmd_wait.hpp"

// This function is used to interperate instructions
// Useful files:
// mlir-aie/lib/Dialect/AIEX/IR/AIEXDialect.cpp
// mlir-aie/lib/Dialect/AIEX/Transforms/AIEDmaToNpu.cpp
// mlir-aie/lib/Targets/AIETargetNPU.cpp

// found in AIETargetNPU.cpp


class npu_sequence{
    public:
        npu_sequence(); // for construct from empty
        npu_sequence(std::vector<uint32_t>& npu_seq); // for construct from vector
        npu_sequence(xrt::bo& bo); // for construct from bo
        npu_sequence(std::string filename); // for construct from file
        void parse_sequence(); // parse the sequence
        // void add_instr(uint32_t instr); // add instruction to the sequence
        void print_sequence(); // print the sequence
        void to_npu();
    private:
        vector<uint32_t> npu_seq;
        std::vector<npu_cmd*> cmds;
        uint32_t npu_rows;
        uint32_t npu_cols;
        uint32_t npu_mem_tile_rows;
        uint32_t npu_minor;
        uint32_t npu_major;
        uint32_t instruction_counts;
        uint32_t instruction_lines;
};

#endif

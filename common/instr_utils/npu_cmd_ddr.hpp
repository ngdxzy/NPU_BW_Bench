#ifndef __NPU_CMD_DDR_HPP__
#define __NPU_CMD_DDR_HPP__

#include "npu_cmd.hpp"


struct npu_ddr_cmd : public npu_cmd{
    uint32_t op_size;
    uint32_t bd_id;
    uint32_t col;
    uint32_t row;
    uint32_t arg_idx;
    uint32_t arg_offset;
    uint32_t constant_0 = 0;
    
    void dump_cmd(uint32_t *bd){
        // Address patch
        this->op_size = bd[1];
        this->col = ((bd[2] >> bd_col_shift) & bd_col_mask);
        this->row = ((bd[2] >> bd_row_shift) & bd_row_mask);
        this->bd_id = ((bd[2] - 0x04) >> bd_id_shift) & bd_id_mask;

        // Buffer descriptor address register address AIEXDialect.cpp line 39

        // argument idx
        this->arg_idx = bd[3];

        // argument offset
        this->arg_offset = bd[4];
    }

    int print_cmd(uint32_t *bd, int line_number, int op_count){
        // Address patch
        MSG_BONDLINE(INSTR_PRINT_WIDTH);
        instr_print(line_number++, bd[0], "DDR patch, OP count: " + std::to_string(op_count)); // AIETargetNPU.cpp line 122
        instr_print(line_number++, bd[1], "Operation size: " + std::to_string(bd[1] / 4)); // AIETargetNPU.cpp line 122

        // Buffer descriptor address register address AIEXDialect.cpp line 39
        instr_print(line_number++, bd[2], "BD register address");
        instr_print(-1, bd[2], "--Location: (row: " + std::to_string(row) + ", col: " + std::to_string(col) + ")");
        instr_print(-1, bd[2], "--BD ID: " + std::to_string(bd_id));

        // argument idx
        instr_print(line_number++, bd[3], "Argument index: " + std::to_string(bd[3]));

        // argument offset
        instr_print(line_number++, bd[4], "Argument offset (Bytes): " + std::to_string(bd[4]));

        // constant 0
        instr_print(line_number++, bd[5], "Constant 0");
        return line_number;
    }

    void to_npu(std::vector<uint32_t>& npu_seq){
        npu_seq.push_back(dma_ddr_patch_write);
        npu_seq.push_back(op_size);
        npu_seq.push_back(
            (col << bd_col_shift) |
            (row << bd_row_shift) |
            (bd_id << bd_id_shift) | 0x1D004
        );
        npu_seq.push_back(arg_idx);
        npu_seq.push_back(arg_offset);
        npu_seq.push_back(constant_0);
    }

    int get_op_lines(){
        return 6;
    }
};

#endif

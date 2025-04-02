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
        this->col = ((bd[6] >> bd_col_shift) & bd_col_mask);
        this->row = ((bd[6] >> bd_row_shift) & bd_row_mask);
        this->bd_id = ((bd[6] - 0x04) >> bd_id_shift) & bd_id_mask;

        // Buffer descriptor address register address AIEXDialect.cpp line 39

        // argument idx
        this->arg_idx = bd[8];

        // argument offset
        this->arg_offset = bd[10];
    }

    int print_cmd(uint32_t *bd, int line_number, int op_count){
        // Address patch
        MSG_BONDLINE(INSTR_PRINT_WIDTH);
        instr_print(line_number++, bd[0], "DDR patch, OP count: " + std::to_string(op_count)); // AIETargetNPU.cpp line 122
        instr_print(line_number++, bd[1], "Operation size: " + std::to_string(this->op_size / 4)); // AIETargetNPU.cpp line 122
        instr_print(line_number++, bd[2], "Useless"); // AIETargetNPU.cpp line 122
        instr_print(line_number++, bd[3], "Useless"); // AIETargetNPU.cpp line 122
        instr_print(line_number++, bd[4], "Useless"); // AIETargetNPU.cpp line 122
        instr_print(line_number++, bd[5], "Useless"); // AIETargetNPU.cpp line 122
        // Buffer descriptor address register address AIEXDialect.cpp line 39
        instr_print(line_number++, bd[6], "BD register address");
        instr_print(-1, bd[6], "--Location: (row: " + std::to_string(row) + ", col: " + std::to_string(col) + ")");
        instr_print(-1, bd[6], "--BD ID: " + std::to_string(bd_id));
        instr_print(line_number++, bd[7], "Useless"); // AIETargetNPU.cpp line 122
        // argument idx
        instr_print(line_number++, bd[8], "Argument index: " + std::to_string(bd[8]));
        instr_print(line_number++, bd[9], "Useless"); // AIETargetNPU.cpp line 122
        // argument offset
        instr_print(line_number++, bd[10], "Argument offset (Bytes): " + std::to_string(bd[10]));
        instr_print(line_number++, bd[11], "Useless"); // AIETargetNPU.cpp line 122
        // constant 0
        // instr_print(line_number++, bd[12], "Constant 0");
        return line_number;
    }

    void to_npu(std::vector<uint32_t>& npu_seq){
        npu_seq.push_back(dma_ddr_patch_write);
        npu_seq.push_back(op_size);
        npu_seq.push_back(0x0);
        npu_seq.push_back(0x0);
        npu_seq.push_back(0x0);
        npu_seq.push_back(0x0);
        npu_seq.push_back(
            (col << bd_col_shift) |
            (row << bd_row_shift) |
            (bd_id << bd_id_shift) | 0x1D004
        );
        npu_seq.push_back(0x0);
        npu_seq.push_back(arg_idx);
        npu_seq.push_back(0x0);
        npu_seq.push_back(arg_offset);
        npu_seq.push_back(0x0);
    }

    int get_op_lines(){
        return 12;
    }
};

#endif

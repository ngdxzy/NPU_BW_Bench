#ifndef __NPU_CMD_WAIT_HPP__
#define __NPU_CMD_WAIT_HPP__

#include "npu_cmd.hpp"


struct npu_wait_cmd : public npu_cmd{
    uint32_t wait_row;
    uint32_t wait_col;
    uint32_t wait_channel;
    uint32_t op_size;
    bool direction; // 0 is S2MM, 1 is MM2S
    void dump_cmd(uint32_t *bd){
        this->op_size = bd[1];
        this->direction = (bd[2] >> wait_sync_direction_shift) & wait_sync_direction_mask;
        this->wait_row = (bd[2] >> wait_sync_row_shift) & wait_sync_row_mask;
        this->wait_col = (bd[2] >> wait_sync_col_shift) & wait_sync_col_mask;
        this->wait_channel = (bd[3] >> wait_sync_channel_shift) & wait_sync_channel_mask;
    }

    int print_cmd(uint32_t *bd, int line_number, int op_count){
        MSG_BONDLINE(INSTR_PRINT_WIDTH);
        instr_print(line_number++, bd[0], "Wait sync, OP count: " + std::to_string(op_count));
        instr_print(line_number++, bd[1], "--Operation size: " + std::to_string(this->op_size));
        instr_print(line_number++, bd[2], "--Location: (row: " + std::to_string(this->wait_row) + ", col: " + std::to_string(this->wait_col) + ")");
        if (this->direction == false){
            instr_print(-1, bd[2], "--S2MM");
        }
        else{
            instr_print(-1, bd[2], "--MM2S");
        }
        instr_print(line_number++, bd[3], "--Channel: " + std::to_string(this->wait_channel));
        return line_number;
    }

    void to_npu(std::vector<uint32_t>& npu_seq){
        npu_seq.push_back(dma_sync_write);
        npu_seq.push_back(this->op_size);
        npu_seq.push_back((this->wait_row << wait_sync_row_shift) | (this->wait_col << wait_sync_col_shift));
        npu_seq.push_back((this->wait_channel << wait_sync_channel_shift) | (1 << wait_sync_row_shift) | (1 << wait_sync_col_shift));
    }

    int get_op_lines(){
        return 4;
    }
};

#endif

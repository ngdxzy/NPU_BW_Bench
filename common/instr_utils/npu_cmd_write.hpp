#ifndef __NPU_CMD_WRITE_HPP__
#define __NPU_CMD_WRITE_HPP__

#include "npu_cmd.hpp"


struct npu_write_cmd : public npu_cmd{
    bool channel_direction; // 0 is S2MM, 1 is MM2S
    uint32_t channel_id;
    uint32_t repeat_count;
    uint32_t reg_addr;
    bool issue_token;
    uint32_t bd_id;
    uint32_t row, col;
    uint32_t op_size;
    uint32_t value;
    bool could_be_push_queue;
    void dump_cmd(uint32_t *bd){    
        this->row = (bd[2] >> bd_row_shift) & bd_row_mask;
        this->col = (bd[2] >> bd_col_shift) & bd_col_mask;
        this->reg_addr = bd[2] & 0xFFFFF;
        if ((this->reg_addr & 0x1FE00) == 0x1d200){
            this->could_be_push_queue = true;
        }
        else{
            this->could_be_push_queue = false;
        }
        if ((reg_addr & 0x10) == 0){
            this->channel_direction = false;
        }
        else{
            this->channel_direction = true;
        }
        if (this->could_be_push_queue){
            this->channel_id = (bd[2] >> queue_channel_shift) & queue_channel_mask;
            this->repeat_count = (bd[4] >> ending_repeat_cnt_shift) & ending_repeat_cnt_mask;
            this->issue_token = (bd[4] >> ending_issue_token_shift) & ending_issue_token_mask;
            this->bd_id = (bd[4] >> ending_bd_id_shift) & ending_bd_id_mask;
        }
        this->value = bd[4];
        this->op_size = bd[5] >> 2;
    }

    int print_cmd(uint32_t *bd, int line_number, int op_count){
        MSG_BONDLINE(INSTR_PRINT_WIDTH);
        instr_print(line_number++, bd[0], "Write 32, OP count: " + std::to_string(op_count));
        instr_print(line_number++, bd[1], "Useless");
        if (this->could_be_push_queue){
                if (this->channel_direction == false){
                    instr_print(line_number++, bd[2], "--S2MM");
                }
            else{
                instr_print(line_number++, bd[2], "--MM2S");
            }
            instr_print(-1, bd[2], "--Location: (row: " + std::to_string(row) + ", col: " + std::to_string(col) + ")");
            instr_print(-1, bd[2], "--Register address: " + std::to_string(this->reg_addr));
            instr_print(-1, bd[2], "--Channel: " + std::to_string(this->channel_id));
            instr_print(line_number++, bd[3], "Useless");
            instr_print(line_number++, bd[4], "--Repeat count: " + std::to_string(this->repeat_count));
            instr_print(-1, bd[4], "--Issue token: " + std::to_string(this->issue_token));
            instr_print(-1, bd[4], "--BD ID: " + std::to_string(this->bd_id));
        }
        else{
            instr_print(line_number++, bd[2], "RTP write");
            instr_print(-1, bd[2], "--Location: (row: " + std::to_string(row) + ", col: " + std::to_string(col) + ")");
            instr_print(-1, bd[2], "--Register address: " + std::to_string(this->reg_addr));
            instr_print(line_number++, bd[3], "Useless");
            instr_print(line_number++, bd[4], "Value: " + std::to_string(this->value));
        }
        instr_print(line_number++, bd[5], "OP size: " + std::to_string(this->op_size));
        return line_number;
    }
    
    void to_npu(std::vector<uint32_t>& npu_seq){
        npu_seq.push_back(queue_write);
        npu_seq.push_back(0x0);
        npu_seq.push_back((this->row << bd_row_shift) + (this->col << bd_col_shift) + (this->reg_addr));
        npu_seq.push_back(0x0);
        npu_seq.push_back(this->value);
        npu_seq.push_back(this->op_size << 2);
    }

    int get_op_lines(){
        return 6;
    }
};

#endif

#ifndef __NPU_CMD_SYNC_HPP__
#define __NPU_CMD_SYNC_HPP__

#include "npu_cmd.hpp"

struct npu_issue_token_cmd : public npu_cmd{
    bool channel_direction; // 0 is S2MM, 1 is MM2S
    uint32_t channel_id;
    uint32_t controller_packet_id;
    uint32_t row, col;
    const uint32_t mask = 0x00000f00;

    void dump_cmd(uint32_t *bd){
        this->row = (bd[1] >> bd_row_shift) & bd_row_mask;
        this->col = (bd[1] >> bd_col_shift) & bd_col_mask;
        if ((bd[1] & 0x10) == 0){
            this->channel_direction = false;
        }
        else{
            this->channel_direction = true;
        }
        this->channel_id = (bd[1] >> queue_channel_shift) & queue_channel_mask;
        this->controller_packet_id = bd[2] >> queue_pkt_id_shift;
    }

    int print_cmd(uint32_t *bd, int line_number, int op_count){ 
        MSG_BONDLINE(INSTR_PRINT_WIDTH);
        instr_print(line_number++, bd[0], "Issue token, OP count: " + std::to_string(op_count));
        if (this->channel_direction == false){
            instr_print(line_number++, bd[1], "--S2MM");
        }
        else{
            instr_print(line_number++, bd[1], "--MM2S");
        }
        instr_print(-1, bd[1], "--Location: (row: " + std::to_string(this->row) + ", col: " + std::to_string(this->col) + ")");
        instr_print(-1, bd[1], "--Channel: " + std::to_string(this->channel_id));
        instr_print(line_number++, bd[2], "Controller packet ID: " + std::to_string(this->controller_packet_id));
        instr_print(line_number++, bd[3], "Mask (constant)");
        return line_number;
    }
    
    void to_npu(std::vector<uint32_t>& npu_seq){
        npu_seq.push_back(dma_issue_token_write);
        if (this->channel_direction == false){
            npu_seq.push_back(0x1D200 + this->channel_id * 0x08 + 0x10 * this->channel_direction + (this->row << bd_row_shift) + (this->col << bd_col_shift));
        }
        else{
            npu_seq.push_back(0x1D200 + this->channel_id * 0x08 + 0x10 * this->channel_direction + (this->row << bd_row_shift) + (this->col << bd_col_shift));
        }
        npu_seq.push_back(this->controller_packet_id << queue_pkt_id_shift);
        npu_seq.push_back(this->mask);
    }

    int get_op_lines(){
        return 4;
    }
};

#endif

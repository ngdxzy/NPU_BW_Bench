#include "npu_instr_utils.hpp"

namespace npu_instr_utils{

void instr_print(int line_number, uint32_t word, std::string msg){
    if (line_number == -1){ // -1 for the case when one line has multiple messages
        MSG_BOX_LINE(INSTR_PRINT_WIDTH, std::dec << std::setw(7) << " | " << std::setw(11) << " | " << msg);
    }
    else{
        MSG_BOX_LINE(INSTR_PRINT_WIDTH, std::dec << std::setw(4) << line_number << " | " << std::hex << std::setfill('0') << std::setw(8) << word << " | " << msg);
    }
}

void unknown_instr(int line_number, uint32_t word){
    instr_print(line_number, word, "\033[31mUnknown instruction\033[0m");
}


void print_bd(xrt::bo& bo){
    bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    vector<uint32_t> bd = vector<uint32_t>(bo);
    // A dma bd starts with a buffer descriptor
    // From mlir-aie/lib/Dialect/AIEX/Transforms/AIEDmaToNpu.cpp
    // Descriptor = ((col & 0xff) << tm.getColumnShift()) | ((row & 0xff) << tm.getRowShift()) | (0x1D000 + bd_id * 0x20);
    // The shifts can be found in mlir-aie/include/aie/Dialect/AIE/IR/AIETargetModel.h
    // We are using AIE2, so check the AIE2TargetModel
    // getColumnShift() = 25
    // getRowShift() = 20
    // In AIE2, there are 16 bds in a shim tile, which means (bd_id * 0x20) is not greater than 0x1E0
    // So, the lower 20 bits of the descriptor is 0x1D000 to 0x1D1E0
    int instr_idx = 0;
    bool is_bd_found = false;

    MSG_BONDLINE(INSTR_PRINT_WIDTH);
    // header, npu information
    instr_print(instr_idx, bd[instr_idx], "NPU information");
    instr_print(-1, bd[instr_idx], "--NPU version: " + std::to_string((bd[instr_idx] >> dev_major_shift) & dev_major_mask) + "." + std::to_string((bd[instr_idx] >> dev_minor_shift) & dev_minor_mask));
    instr_print(-1, bd[instr_idx], "--NPU rows: " + std::to_string((bd[instr_idx] >> dev_n_row_shift) & dev_n_row_mask));
    instr_idx++;
    instr_print(instr_idx, bd[instr_idx], "--NPU cols: " + std::to_string((bd[instr_idx] >> dev_num_cols_shift) & dev_num_cols_mask));
    instr_print(-1, bd[instr_idx], "--NPU memory tile rows: " + std::to_string((bd[instr_idx] >> dev_mem_tile_rows_shift) & dev_mem_tile_rows_mask));
    instr_idx++;

    // Instruction commands
    instr_print(instr_idx, bd[instr_idx], "Instruction commands: " + std::to_string(bd[instr_idx]));
    instr_idx++;

    // Instruction lines
    instr_print(instr_idx, bd[instr_idx], "Instruction lines: " + std::to_string(bd[instr_idx] / 4));
    instr_idx++;
    int op_count = 0;
    while (instr_idx < bd.size()){
        // Check if the instruction is a dma bd
        if (bd[instr_idx] == op_headers::dma_block_write){
            is_bd_found = true;
            MSG_BONDLINE(INSTR_PRINT_WIDTH);
            // This is a dma bd
            uint32_t word = bd[instr_idx];
            instr_print(instr_idx, word, "DMA block write, OP count: " + std::to_string(++op_count));
            instr_idx++;
            // word 0:
            word = bd[instr_idx];
            uint32_t col = ((word >> bd_col_shift) & bd_col_mask);
            uint32_t row = ((word >> bd_row_shift) & bd_row_mask);
            uint32_t bd_id = ((word >> bd_id_shift) & bd_id_mask);
            instr_print(instr_idx, word, "--Location: (row: " + std::to_string(row) + ", col: " + std::to_string(col) + ")");
            instr_print(-1, word, "--BD ID: " + std::to_string(bd_id));
            instr_idx++;

            // all words can be found in mlir-aie/lib/Dialect/AIEX/Transforms/AIEDmaToNpu.cpp close to line 577
            // word 0: unknown yet
            word = bd[instr_idx];
            instr_print(instr_idx, word, "Operation size: " + std::to_string(word / 4));
            instr_idx++;
            // word 1: Buffer length
            word = bd[instr_idx];
            uint32_t buffer_length = word;
            instr_print(instr_idx, word, "--Buffer length: " + size_t_to_string(buffer_length));
            instr_idx++;

            // word 2: Buffer offset
            word = bd[instr_idx];
            uint32_t buffer_offset = word;
            instr_print(instr_idx, word, "--Buffer offset: " + size_t_to_string(buffer_offset));
            instr_idx++;

            // word 3: Packet information
            word = bd[instr_idx];
            if ((word >> en_packet_shift) & en_packet_mask){
                // This is a packet
                instr_print(instr_idx, word, "--Packet enabled");
                instr_print(-1, word, "--Out of order id: " + std::to_string((word >> out_of_order_shift) & out_of_order_mask));
                instr_print(-1, word, "--Packet id: " + std::to_string((word >> packet_id_shift) & packet_id_mask));
                instr_print(-1, word, "--Packet type: " + std::to_string((word >> packet_type_shift) & packet_type_mask));
            }
            else{
                instr_print(instr_idx, word, "Packet disabled");
            }
            instr_idx++;

            // word 4: D0
            uint32_t d0_size, d0_stride;
            word = bd[instr_idx];
            if (word == 0){
                instr_print(instr_idx, word, "A linear transfer, no D0");
            }
            else{
                d0_size = (word >> dim_size_shift) & dim_size_mask;
                d0_stride = (word >> dim_stride_shift) & dim_stride_mask;
                d0_stride += 1; // The saved value is the stride - 1
                instr_print(instr_idx, word, "--D0 size, stride: " + size_t_to_string(d0_size) + ", " + size_t_to_string(d0_stride));
            }
            instr_idx++;

            // word 5: D1
            uint32_t d1_size, d1_stride;
            word = bd[instr_idx];
            d1_size = (word >> dim_size_shift) & dim_size_mask;
            if (d1_size == 0){
                instr_print(instr_idx, word, "--No D1");
            }
            else{
                d1_stride = (word >> dim_stride_shift) & dim_stride_mask;
                d1_stride += 1; // The saved value is the stride - 1
                instr_print(instr_idx, word, "--D1 size, stride: " + size_t_to_string(d1_size) + ", " + size_t_to_string(d1_stride));
            }
            instr_idx++;
            // word 6: D2
            uint32_t d2_size, d2_stride;
            word = bd[instr_idx];
            if (word == 0){
                instr_print(instr_idx, word, "--No D2");
            }
            else{
                d2_stride = (word >> dim_stride_shift) & dim_stride_mask;
                d2_stride += 1; // The saved value is the stride - 1
                d2_size = buffer_length / (d0_size * d1_size);
                instr_print(instr_idx, word, "--D2 stride: " + size_t_to_string(d2_stride));
                instr_print(-1, word, "--Inferred D2 size: " + size_t_to_string(d2_size));
            }
            instr_idx++;
            // word 7: D3, Iteration dimension
            uint32_t curr_iter, iter_size, iter_stride;
            word = bd[instr_idx];
            if (word == 0){
                instr_print(instr_idx, word, "--No Iteration dimension");
            }
            else{
                curr_iter = (word >> curr_iter_shift) & curr_iter_mask;
                iter_size = (word >> iter_size_shift) & iter_size_mask;
                iter_stride = (word >> iter_stride_shift) & iter_stride_mask;
                iter_stride += 1; // The saved value is the stride - 1
                iter_size += 1; // The saved value is the size - 1
                instr_print(instr_idx, word, "--Current iteration: " + size_t_to_string(curr_iter));
                instr_print(-1, word, "--Iteration size: " + size_t_to_string(iter_size));
                instr_print(-1, word, "--Iteration stride: " + size_t_to_string(iter_stride));
            }
            instr_idx++;
            // word 8: Next BD, Lock information
            word = bd[instr_idx];
            uint32_t next_bd_id = (word >> next_bd_id_shift) & next_bd_id_mask;
            uint32_t valid_bd = (word >> valid_bd_shift) & valid_bd_mask;
            // These informantion are provided but not used on NPU2
            // uint32_t lock_rel_val = (word >> get_lock_rel_val_shift) & get_lock_rel_val_mask;
            // uint32_t lock_rel_id = (word >> get_lock_rel_id_shift) & get_lock_rel_id_mask;
            // uint32_t lock_acq_enable = (word >> get_lock_acq_enable_shift) & get_lock_acq_enable_mask;
            // uint32_t lock_acq_val = (word >> get_lock_acq_val_shift) & get_lock_acq_val_mask;
            // uint32_t lock_acq_id = (word >> get_lock_acq_id_shift) & get_lock_acq_id_mask;
            instr_print(instr_idx, word, "--Next BD ID: " + std::to_string(next_bd_id));
            instr_print(-1, word, "--Valid BD: " + std::to_string(valid_bd));
            // MSG_BOX_LINE(40, "--Lock relative value: " << size_t_to_string(lock_rel_val));
            // MSG_BOX_LINE(40, "--Lock relative id: " << size_t_to_string(lock_rel_id));
            // MSG_BOX_LINE(40, "--Lock acquire enable: " << lock_acq_enable);
            // MSG_BOX_LINE(40, "--Lock acquire value: " << size_t_to_string(lock_acq_val));
            // MSG_BOX_LINE(40, "--Lock acquire id: " << size_t_to_string(lock_acq_id));
            instr_idx++;
        }
        else if (bd[instr_idx] == op_headers::dma_ddr_patch_write){
            // Address patch
            MSG_BONDLINE(INSTR_PRINT_WIDTH);
            uint32_t word = bd[instr_idx];
            instr_print(instr_idx, word, "DDR patch, OP count: " + std::to_string(++op_count)); // AIETargetNPU.cpp line 122
            instr_idx++;
            word = bd[instr_idx];
            instr_print(instr_idx, word, "Operation size: " + std::to_string(word / 4)); // AIETargetNPU.cpp line 122
            instr_idx++;

            // Buffer descriptor address register address AIEXDialect.cpp line 39
            word = bd[instr_idx];
            instr_print(instr_idx, word, "BD register address");
            uint32_t col = ((word >> bd_col_shift) & bd_col_mask);
            uint32_t row = ((word >> bd_row_shift) & bd_row_mask);
            instr_print(-1, word, "--Location: (row: " + std::to_string(row) + ", col: " + std::to_string(col) + ")");
            int bd_id = ((word - 0x04) >> bd_id_shift) & bd_id_mask;
            instr_print(-1, word, "--BD ID: " + std::to_string(bd_id));
            instr_idx++;

            // argument idx
            word = bd[instr_idx];
            instr_print(instr_idx, word, "Argument index: " + std::to_string(word));
            instr_idx++;

            // argument offset
            word = bd[instr_idx];
            instr_print(instr_idx, word, "Argument offset (Bytes): " + std::to_string(word));
            instr_idx++;

            // constant 0
            word = bd[instr_idx];
            instr_print(instr_idx, word, "Constant 0");
            instr_idx++;
        }
        else if (bd[instr_idx] == op_headers::dma_issue_token_write){
            MSG_BONDLINE(INSTR_PRINT_WIDTH);
            uint32_t word = bd[instr_idx];

            instr_print(instr_idx, word, "Issue token, OP count: " + std::to_string(++op_count));
            instr_idx++;
            word = bd[instr_idx];
            if ((word & 0x10) == 0){
                instr_print(instr_idx, word, "--S2MM");

            }
            else{
                instr_print(instr_idx, word, "--MM2S");
            }
            instr_print(instr_idx, word, "--Channel: " + std::to_string((word >> queue_channel_shift) & queue_channel_mask));
            instr_idx++;
            word = bd[instr_idx];
            instr_print(instr_idx, word, "Controller packet ID: " + std::to_string((word >> queue_pkt_id_shift) & queue_pkt_id_mask));
            instr_idx++;
            word = bd[instr_idx];
            instr_print(instr_idx, word, "Mask (constant)");
            instr_idx++;
        }
        else if (bd[instr_idx] == op_headers::queue_write){
            MSG_BONDLINE(INSTR_PRINT_WIDTH);
            uint32_t word = bd[instr_idx];
            instr_print(instr_idx, bd[instr_idx], "Queue bd, OP count: " + std::to_string(++op_count)); // AIETargetNPU.cpp line 70
            instr_idx++;
            word = bd[instr_idx];
            if ((word & 0x10) == 0){
                instr_print(instr_idx, word, "--S2MM");
            }
            else{
                instr_print(instr_idx, word, "--MM2S");
            }
            instr_print(-1, word, "--Channel: " + std::to_string((word >> queue_channel_shift) & queue_channel_mask));
            instr_idx++;

            word = bd[instr_idx];
            instr_print(instr_idx, word, "--BD ID: " + std::to_string((word >> ending_bd_id_shift) & ending_bd_id_mask));
            instr_print(-1, word, "--Repeat count: " + std::to_string((word >> ending_repeat_cnt_shift) & ending_repeat_cnt_mask));
            instr_print(-1, word, "--Issue token: " + std::to_string((word >> ending_issue_token_shift) & ending_issue_token_mask));
            instr_idx++;
        }
        else if (bd[instr_idx] == op_headers::dma_sync_write){ // Wait sync, AIETargetNPU.cpp line 62
            MSG_BONDLINE(INSTR_PRINT_WIDTH);
            instr_print(instr_idx, bd[instr_idx], "Wait sync: TXN_OPC_TCT, OP count: " + std::to_string(++op_count));
            instr_idx++;
            uint32_t word = bd[instr_idx];
            instr_print(instr_idx, word, "--Operation size: " + std::to_string(word / 4));
            instr_idx++;
            word = bd[instr_idx];
            int row = (word >> wait_sync_row_shift) & wait_sync_row_mask;
            int col = (word >> wait_sync_col_shift) & wait_sync_col_mask;
            instr_print(instr_idx, word, "--Location: (row: " + std::to_string(row) + ", col: " + std::to_string(col) + ")");
            instr_idx++;
            word = bd[instr_idx];
            row = (word >> wait_sync_row_shift) & wait_sync_row_mask;
            col = (word >> wait_sync_col_shift) & wait_sync_col_mask;
            int channel = (word >> wait_sync_channel_shift) & wait_sync_channel_mask;
            instr_print(instr_idx, bd[instr_idx], "--Useless: (num rows: " + std::to_string(row) + ", num cols: " + std::to_string(col) + ")");
            instr_print(-1, bd[instr_idx], "--Channel: " + std::to_string(channel));
            instr_idx++;
        }
        else{
            unknown_instr(instr_idx, bd[instr_idx]);
            instr_idx += 1;
        }
        if (instr_idx >= bd.size()){
            break;
        }
    }
    if (!is_bd_found){
        MSG_BONDLINE(INSTR_PRINT_WIDTH);
        MSG_BOX_LINE(INSTR_PRINT_WIDTH, "No Operation found!");
    }
    MSG_BONDLINE(INSTR_PRINT_WIDTH);
}

} // end namespace
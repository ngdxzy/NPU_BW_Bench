#ifndef __NPU_CMD_HPP__
#define __NPU_CMD_HPP__

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

const int INSTR_PRINT_WIDTH = 80;

typedef enum{
    queue_write = 0x00,
    dma_block_write = 0x01,
    dma_issue_token_write = 0x03,
    dma_sync_write = 0x80,
    dma_ddr_patch_write = 0x81,
} op_headers;

typedef enum{
    dev_n_row_mask = 0xFF,
    dev_gen_mask = 0xFF,
    dev_minor_mask = 0xFF,
    dev_major_mask = 0xFF,
    dev_mem_tile_rows_mask = 0xFF,
    dev_num_cols_mask = 0xFF,
    is_bd_mask = 0xFF,
    bd_col_mask = 0x7F,
    bd_row_mask = 0x1F,
    bd_id_mask = 0xF,
    en_packet_mask = 0x1,
    out_of_order_mask = 0x3F,
    packet_id_mask = 0x1F,
    packet_type_mask = 0x7,
    dim_size_mask = 0x3FF,
    dim_stride_mask = 0xFFFFF,
    curr_iter_mask = 0x3FF,
    iter_size_mask = 0x3FF,
    iter_stride_mask = 0xFFFFF,
    next_bd_id_mask = 0xF,
    use_next_bd_mask = 0x1,
    valid_bd_mask = 0x1,
    get_lock_rel_val_mask = 0xEF,
    get_lock_rel_id_mask = 0xF,
    get_lock_acq_enable_mask = 0x1,
    get_lock_acq_val_mask = 0xEF,
    get_lock_acq_id_mask = 0xF,
    queue_channel_mask = 0x1,
    queue_pkt_id_mask = 0xFFFFFF,
    ending_bd_id_mask = 0xF,
    ending_repeat_cnt_mask = 0xFF,
    ending_issue_token_mask = 0x1,
    wait_sync_row_mask = 0xFF,
    wait_sync_col_mask = 0xFF,
    wait_sync_channel_mask = 0xFF,
    wait_sync_direction_mask = 0x1,
} npu_instr_mask;

typedef enum{
    dev_n_row_shift = 24,
    dev_gen_shift = 16,
    dev_minor_shift = 8,
    dev_major_shift = 0,
    dev_mem_tile_rows_shift = 8,
    dev_num_cols_shift = 0,
    is_bd_shift = 12,
    bd_col_shift = 25,
    bd_row_shift = 20,
    bd_id_shift = 5,
    en_packet_shift = 30,
    out_of_order_shift = 24,
    packet_id_shift = 19,
    packet_type_shift = 16, 
    dim_size_shift = 20,
    dim_stride_shift = 0,
    curr_iter_shift = 26,
    iter_size_shift = 20,
    iter_stride_shift = 0,
    next_bd_id_shift = 27,
    use_next_bd_shift = 26,
    valid_bd_shift = 25,
    get_lock_rel_val_shift = 18,
    get_lock_rel_id_shift = 13,
    get_lock_acq_enable_shift = 12, 
    get_lock_acq_val_shift = 5,
    get_lock_acq_id_shift = 0,
    queue_channel_shift = 3,
    queue_pkt_id_shift = 8,
    ending_bd_id_shift = 0,
    ending_repeat_cnt_shift = 16,
    ending_issue_token_shift = 31,
    wait_sync_row_shift = 8,
    wait_sync_col_shift = 16,
    wait_sync_channel_shift = 24,
    wait_sync_direction_shift = 0,
} npu_instr_shifts;

typedef enum{
    npu_sync,
    npu_dma_block,
    npu_queue,
    npu_wait,
    npu_write,
    npu_arg_set
} npu_cmd_type;

inline void instr_print(int line_number, uint32_t word, std::string msg){
    if (line_number == -1){ // -1 for the case when one line has multiple messages
        MSG_BOX_LINE(INSTR_PRINT_WIDTH, std::dec << std::setw(7) << " | " << std::setw(11) << " | " << msg);
    }
    else{
        MSG_BOX_LINE(INSTR_PRINT_WIDTH, std::dec << std::setw(4) << line_number << " | " << std::hex << std::setfill('0') << std::setw(8) << word << " | " << msg);
    }
}

struct npu_cmd{
    virtual int print_cmd(uint32_t *bd, int line_number, int op_count) = 0;
    virtual void to_npu(std::vector<uint32_t>& npu_seq) = 0;
    // virtual npu_cmd_type get_type() = 0;
    virtual void dump_cmd(uint32_t *bd) = 0;
    virtual int get_op_lines() = 0;
};


#endif

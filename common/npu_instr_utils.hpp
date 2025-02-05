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

// This function is used to interperate instructions
// Useful files:
// mlir-aie/lib/Dialect/AIEX/IR/AIEXDialect.cpp
// mlir-aie/lib/Dialect/AIEX/Transforms/AIEDmaToNpu.cpp
// mlir-aie/lib/Targets/AIETargetNPU.cpp

// found in AIETargetNPU.cpp
namespace npu_instr_utils{
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
} npu_instr_shifts;

const int INSTR_PRINT_WIDTH = 80;

void instr_print(int line_number, uint32_t word, std::string msg);

void unknown_instr(int line_number, uint32_t word);

void print_bd(xrt::bo& bo);

} // end namespace
#endif

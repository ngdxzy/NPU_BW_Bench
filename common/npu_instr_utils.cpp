#include "npu_instr_utils.hpp"


npu_sequence::npu_sequence(std::vector<uint32_t>& npu_seq){
    this->npu_seq.acquire(npu_seq.size());
    this->npu_seq.copy_from(npu_seq);

    // Parse the npu sequence
    this->parse_sequence();
}

npu_sequence::npu_sequence(xrt::bo& bo){
    this->npu_seq = vector<uint32_t>(bo);

    // Parse the npu sequence
    this->parse_sequence();
}

npu_sequence::npu_sequence(std::string filename){
    std::ifstream file(filename);
    std::string line;
    std::vector<uint32_t> npu_seq;
    while (std::getline(file, line)){
        npu_seq.push_back(std::stoi(line));
    }
    this->npu_seq.acquire(npu_seq.size());
    this->npu_seq.copy_from(npu_seq);

    // Parse the npu sequence   
    this->parse_sequence();
}

npu_sequence::npu_sequence(){
    // Empty constructor
}

void npu_sequence::parse_sequence(){
    // Parse the npu sequence
    this->npu_major = (this->npu_seq[0] >> dev_major_shift) & dev_major_mask;
    this->npu_minor = (this->npu_seq[0] >> dev_minor_shift) & dev_minor_mask;
    this->npu_rows = (this->npu_seq[0] >> dev_n_row_shift) & dev_n_row_mask;
    this->npu_cols = (this->npu_seq[1] >> dev_num_cols_shift) & dev_num_cols_mask;
    this->npu_mem_tile_rows = (this->npu_seq[1] >> dev_mem_tile_rows_shift) & dev_mem_tile_rows_mask;
    this->instruction_counts = this->npu_seq[2];
    this->instruction_lines = this->npu_seq[3] / 4;
    int i = 4;
    while (i < this->npu_seq.size()){
        if (this->npu_seq[i] == op_headers::dma_block_write){
            LOG_VERBOSE(1, "DMA block write");
            npu_cmd* cmd = new npu_dma_block_cmd();
            cmd->dump_cmd(this->npu_seq.data() + i);
            this->cmds.push_back(cmd);
            i += cmd->get_op_lines();
        }
        else if (this->npu_seq[i] == op_headers::dma_ddr_patch_write){
            LOG_VERBOSE(1, "DMA DDR patch write");
            npu_cmd* cmd = new npu_ddr_cmd();
            cmd->dump_cmd(&(this->npu_seq[i]));
            this->cmds.push_back(cmd);
            i += cmd->get_op_lines();
        }
        else if (this->npu_seq[i] == op_headers::dma_issue_token_write){
            LOG_VERBOSE(1, "DMA issue token write");
            npu_cmd* cmd = new npu_issue_token_cmd();
            cmd->dump_cmd(&(this->npu_seq[i]));
            this->cmds.push_back(cmd);
            i += cmd->get_op_lines();
        }
        else if (this->npu_seq[i] == op_headers::queue_write){
            LOG_VERBOSE(1, "Queue write");
            npu_cmd* cmd = new npu_queue_cmd();
            cmd->dump_cmd(&(this->npu_seq[i]));
            this->cmds.push_back(cmd);
            i += cmd->get_op_lines();
        }
        else if (this->npu_seq[i] == op_headers::dma_sync_write){ // Wait sync, AIETargetNPU.cpp line 62
            LOG_VERBOSE(1, "DMA sync write");
            npu_cmd* cmd = new npu_wait_cmd();
            cmd->dump_cmd(&(this->npu_seq[i]));
            this->cmds.push_back(cmd);
            i += cmd->get_op_lines();
        }
        else{
            i++;
        }
    }
}

void npu_sequence::print_sequence(){
    int line_number = 0;
    MSG_BONDLINE(INSTR_PRINT_WIDTH);
    instr_print(line_number, this->npu_seq[line_number], "NPU information");
    instr_print(-1, this->npu_seq[line_number], "--NPU version: " + std::to_string(this->npu_major) + "." + std::to_string(this->npu_minor));
    instr_print(-1, this->npu_seq[line_number], "--NPU rows: " + std::to_string(this->npu_rows));
    line_number++;
    instr_print(line_number, this->npu_seq[line_number], "--NPU cols: " + std::to_string(this->npu_cols));
    instr_print(-1, this->npu_seq[line_number], "--NPU memory tile rows: " + std::to_string(this->npu_mem_tile_rows));
    line_number++;

    // Instruction commands
    instr_print(line_number, this->npu_seq[line_number], "Instruction commands: " + std::to_string(this->npu_seq[line_number]));
    line_number++;

    // Instruction lines
    instr_print(line_number, this->npu_seq[line_number], "Instruction lines: " + std::to_string(this->npu_seq[line_number] / 4));
    line_number++;

    for (int i = 0; i < this->cmds.size(); i++){
        line_number = this->cmds[i]->print_cmd(&(this->npu_seq[line_number]), line_number, i);
    }
    MSG_BONDLINE(INSTR_PRINT_WIDTH);
}

void npu_sequence::to_npu(){
    std::vector<uint32_t> npu_seq;
    
    npu_seq.push_back(
        (this->npu_major << dev_major_shift) |
        (this->npu_minor << dev_minor_shift) |
        (this->npu_rows << dev_n_row_shift)  | 0x00030000
    );
    
    npu_seq.push_back(
        (this->npu_cols << dev_num_cols_shift) |
        (this->npu_mem_tile_rows << dev_mem_tile_rows_shift)
    );
    npu_seq.push_back(this->instruction_counts);
    npu_seq.push_back(this->instruction_lines * 4);
    for (int i = 0; i < this->cmds.size(); i++){
        this->cmds[i]->to_npu(npu_seq);
    }
    
    for (int i = 0; i < this->npu_seq.size(); i++){
        if (npu_seq[i] != this->npu_seq[i]){
            std::cout << std::dec << std::setw(2) << i << " " << std::hex << std::right << std::setfill('0') << std::setw(8)  << npu_seq[i] << " " << std::hex << std::right << std::setfill('0') << std::setw(8)  << this->npu_seq[i]<< std::endl;
        }
    }
}

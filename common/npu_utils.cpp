#include "npu_utils.hpp"


// global device, used by all npu_app instances, only one instance is allowed

npu_app::npu_app(int max_xclbins, int max_instrs, unsigned int device_id){
    this->device = xrt::device(device_id);
    this->kernel_descs.resize(max_xclbins);
    this->hw_descs.resize(max_instrs);
    this->registered_xclbin_names.clear();
    this->kernel_desc_count = 0;
    this->hw_desc_count = 0;
}

int npu_app::register_accel_app(accel_user_desc& user_desc){
    int xclbin_id = -1;
    for (int i = 0; i < this->registered_xclbin_names.size(); i++){
        if (this->registered_xclbin_names[i] == user_desc.xclbin_name){
            xclbin_id = i;
            break;
        }
    }
    if (xclbin_id == -1){ // the xclbin is not registered yet
        if (this->kernel_desc_count >= this->kernel_descs.size()){
            throw std::runtime_error("Max number of xclbins reached");
        }
        if (_load_xclbin(user_desc.xclbin_name) != 0){
            std::cout<< "Load " << user_desc.xclbin_name << "ERROR!" << std::endl;
            exit(-1);
        }
        this->registered_xclbin_names.push_back(user_desc.xclbin_name);
        xclbin_id = this->registered_xclbin_names.size() - 1;
    }
    // register the instr
    int app_id = -1;
    for (int i = 0; i < this->hw_descs.size(); i++){
        if (this->hw_descs[i].instr_name == user_desc.instr_name){
            app_id = i;
            break;
        }
    }
    if (app_id == -1){ // instr is not registered yet
        if (this->hw_desc_count >= this->hw_descs.size()){
            throw std::runtime_error("Max number of instructions reached");
        }
        this->hw_descs[this->hw_desc_count].kernel_desc = &(this->kernel_descs[xclbin_id]);
        _load_instr_sequence(user_desc, this->hw_descs[this->hw_desc_count]);
        app_id = this->hw_desc_count;
        this->hw_desc_count++;
    }
    return app_id;
}


int npu_app::_load_instr_sequence(accel_user_desc& user_desc, accel_hw_desc& hw_desc){
    std::ifstream instr_file(user_desc.instr_name);
    hw_desc.instr_name = user_desc.instr_name;
    std::string line;
    std::vector<uint32_t> instr_v;
    while (std::getline(instr_file, line)) {
        std::istringstream iss(line);
        uint32_t a;
        if (!(iss >> std::hex >> a)) {
            throw std::runtime_error("Unable to parse instruction file\n");
        }
        instr_v.push_back(a);
    }
    hw_desc.bo_instr = xrt::bo(this->device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, hw_desc.kernel_desc->kernel.group_id(1));
    void *bufInstr = hw_desc.bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
    hw_desc.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    hw_desc.instr_size = instr_v.size();
    return 0;
}


int npu_app::_load_xclbin(std::string xclbin_name){
    this->kernel_descs[this->kernel_desc_count].xclbin = xrt::xclbin(xclbin_name);
    int verbosity = 0;
    std::string Node = "MLIR_AIE";
    auto xkernels = this->kernel_descs[this->kernel_desc_count].xclbin.get_kernels();
    auto xkernel = *std::find_if(
        xkernels.begin(), 
        xkernels.end(),
        [Node, verbosity](xrt::xclbin::kernel &k) {
            auto name = k.get_name();
            if (verbosity >= 1) {
            std::cout << "Name: " << name << std::endl;
            }
            return name.rfind(Node, 0) == 0;
        }
    );
    this->device.register_xclbin(this->kernel_descs[this->kernel_desc_count].xclbin);
    // std::cout << "Registering xclbin: " << xclbin_name << "\n";
    auto kernelName = xkernel.get_name();
    this->kernel_descs[this->kernel_desc_count].context = xrt::hw_context(this->device, this->kernel_descs[this->kernel_desc_count].xclbin.get_uuid());
    this->kernel_descs[this->kernel_desc_count].kernel = xrt::kernel(this->kernel_descs[this->kernel_desc_count].context, kernelName);
    this->kernel_desc_count++;
    return 0;
}

xrt::bo npu_app::create_buffer(size_t size, int group_id, int app_id){
    // std::cout << "Group ID: " << (group_id) << std::endl;
    if (app_id >= this->hw_descs.size()){
        throw std::runtime_error("App ID is out of range");
    }
    return xrt::bo(this->device, size, XRT_BO_FLAGS_HOST_ONLY, this->hw_descs[app_id].kernel_desc->kernel.group_id(group_id));
}

ert_cmd_state npu_app::run(xrt::bo& In0, xrt::bo& In1, xrt::bo& Out0, xrt::bo& Out1, int app_id){
    unsigned int opcode = 3;
    auto run = this->hw_descs[app_id].kernel_desc->kernel(opcode, this->hw_descs[app_id].bo_instr, this->hw_descs[app_id].instr_size, In0, In1, Out0, Out1);
    ert_cmd_state r = run.wait();
    return r;
}

ert_cmd_state npu_app::run(xrt::bo& In0, xrt::bo& In1, xrt::bo& Out0, int app_id){
    unsigned int opcode = 3;
    auto run = this->hw_descs[app_id].kernel_desc->kernel(opcode, this->hw_descs[app_id].bo_instr, this->hw_descs[app_id].instr_size, In0, In1, Out0);

    ert_cmd_state r = run.wait();
    return r;
}

ert_cmd_state npu_app::run(xrt::bo& In0, xrt::bo& Out0, int app_id){
    unsigned int opcode = 3;
    auto run = this->hw_descs[app_id].kernel_desc->kernel(opcode, this->hw_descs[app_id].bo_instr, this->hw_descs[app_id].instr_size, In0, Out0);

    ert_cmd_state r = run.wait();
    return r;
}

npu_app::~npu_app(){
    // std::cout<<"clear bin!" << std::endl;
    // this->kernel.~kernel();
    // this->bo_instr.~bo();
    // this->context.~hw_context();
}

void npu_app::list_kernels(){
    std::cout << "Listing kernels: (Total: " << this->hw_descs.size() << ")" << std::endl;
    for (int i = 0; i < this->hw_descs.size(); i++){
        std::cout << "Instruction: " << this->hw_descs[i].instr_name << std::endl;
    }
    std::cout << "Listing xclbins: (Total: " << this->kernel_descs.size() << ")" << std::endl;
    for (int i = 0; i < this->kernel_descs.size(); i++){
        std::cout << "Xclbin: " << &this->kernel_descs[i].xclbin << std::endl;
    }
}
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
    LOG_VERBOSE_IF_ELSE(2, xclbin_id > -1, 
        "Found xclbin: " << user_desc.xclbin_name << "registered as id " << xclbin_id << "!",
        "Xclbin: " << user_desc.xclbin_name << " not registered yet!"
    );

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
        LOG_VERBOSE(2, "Xclbin: " << user_desc.xclbin_name << " registered as id " << xclbin_id << "!");
        this->kernel_desc_count++;
    }
    // register the instr
    int app_id = -1;
    for (int i = 0; i < this->hw_descs.size(); i++){
        if (this->hw_descs[i].instr_name == user_desc.instr_name){
            app_id = i;
            break;
        }
    }
    LOG_VERBOSE_IF_ELSE(2, app_id > -1, 
        "Found instruction: " << user_desc.instr_name << "registered as id " << app_id << "!",
        "Instruction: " << user_desc.instr_name << " not registered yet!"
    );
    if (app_id == -1){ // instr is not registered yet
        if (this->hw_desc_count >= this->hw_descs.size()){
            throw std::runtime_error("Max number of instructions reached");
        }
        this->hw_descs[this->hw_desc_count].kernel_desc = &(this->kernel_descs[xclbin_id]);
        _load_instr_sequence(user_desc, this->hw_descs[this->hw_desc_count]);
        app_id = this->hw_desc_count;
        LOG_VERBOSE(2, "Instruction: " << user_desc.instr_name << " registered as id " << app_id << "!");
        this->hw_desc_count++;
    }
    return app_id;
}


int npu_app::_load_instr_sequence(accel_user_desc& user_desc, accel_hw_desc& hw_desc){
    LOG_VERBOSE(2, "Loading instruction sequence: " << user_desc.instr_name);
    std::ifstream instr_file(user_desc.instr_name, std::ios::binary);
    hw_desc.instr_name = user_desc.instr_name;
#define INSTR_IN_BIN 1
#if INSTR_IN_BIN
    instr_file.seekg(0, std::ios::end);
    size_t instr_size = instr_file.tellg();
    instr_file.seekg(0, std::ios::beg);

    if (instr_size % 4 != 0){
        throw std::runtime_error("Instr file is invalied!");
    }

    std::vector<uint32_t> instr_v(instr_size / 4);
    if (!instr_file.read(reinterpret_cast<char *>(instr_v.data()), instr_size)) {
            throw std::runtime_error("Failed to read instruction file\n");
    }
#else
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
#endif
    hw_desc.bo_instr = xrt::bo(this->device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, hw_desc.kernel_desc->kernel.group_id(1));
    void *bufInstr = hw_desc.bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
    hw_desc.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    hw_desc.instr_size = instr_v.size();
    LOG_VERBOSE(2, "Instruction sequence loaded successfully!");
    return 0;
}


int npu_app::_load_xclbin(std::string xclbin_name){
    LOG_VERBOSE(2, "Loading xclbin: " << xclbin_name);
    this->kernel_descs[this->kernel_desc_count].xclbin = xrt::xclbin(xclbin_name);
    // int verbosity = VERBOSE;
    std::string Node = "MLIR_AIE";
    auto xkernels = this->kernel_descs[this->kernel_desc_count].xclbin.get_kernels();
    auto xkernel = *std::find_if(
        xkernels.begin(), 
        xkernels.end(),
        [Node](xrt::xclbin::kernel &k) {
            auto name = k.get_name();
            return name.rfind(Node, 0) == 0;
        }
    );
    this->device.register_xclbin(this->kernel_descs[this->kernel_desc_count].xclbin);
    auto kernelName = xkernel.get_name();
    std::map<std::string, uint32_t> qos_map = {
        {"gops", 100000},
        {"dma_bandwidth", 180},
        {"priority", AMDXDNA_QOS_HIGH_PRIORITY}
    };
    this->kernel_descs[this->kernel_desc_count].context = xrt::hw_context(this->device, this->kernel_descs[this->kernel_desc_count].xclbin.get_uuid(), qos_map);
    this->kernel_descs[this->kernel_desc_count].context;
    this->kernel_descs[this->kernel_desc_count].kernel = xrt::kernel(this->kernel_descs[this->kernel_desc_count].context, kernelName);
    LOG_VERBOSE(2, "Xclbin: " << xclbin_name << " loaded successfully!");
    return 0;
}

xrt::bo npu_app::create_buffer(size_t size, int group_id, int app_id){
    LOG_VERBOSE(2, "Creating buffer with size: " << size << " and group_id: " << group_id << " and app_id: " << app_id);
    if (app_id >= this->hw_descs.size()){
        throw std::runtime_error("App ID is out of range");
    }
    return xrt::bo(this->device, size, XRT_BO_FLAGS_HOST_ONLY, this->hw_descs[app_id].kernel_desc->kernel.group_id(group_id));
}

template<typename T>
buffer<T> npu_app::create_bo_buffer(size_t size, int group_id, int app_id){
    LOG_VERBOSE(2, "Creating buffer buffer with size: " << size << " and group_id: " << group_id << " and app_id: " << app_id);
    return buffer<T>(size, this->device, this->hw_descs[app_id].kernel_desc->kernel, group_id);
}

template buffer<float> npu_app::create_bo_buffer<float>(size_t size, int group_id, int app_id);
template buffer<uint32_t> npu_app::create_bo_buffer<uint32_t>(size_t size, int group_id, int app_id);
template buffer<int32_t> npu_app::create_bo_buffer<int32_t>(size_t size, int group_id, int app_id);
template buffer<int8_t> npu_app::create_bo_buffer<int8_t>(size_t size, int group_id, int app_id);
template buffer<std::bfloat16_t> npu_app::create_bo_buffer<std::bfloat16_t>(size_t size, int group_id, int app_id);

ert_cmd_state npu_app::run(xrt::bo& In0, xrt::bo& In1, xrt::bo& Out0, xrt::bo& Out1, int app_id){
    unsigned int opcode = 3;
    LOG_VERBOSE(3, "Running kernel with app_id: " << app_id);
    auto run = this->hw_descs[app_id].kernel_desc->kernel(opcode, this->hw_descs[app_id].bo_instr, this->hw_descs[app_id].instr_size, In0, In1, Out0, Out1);
    ert_cmd_state r = run.wait();
    LOG_VERBOSE(3, "Kernel run finished with status: " << r);
    return r;
}

ert_cmd_state npu_app::run(xrt::bo& In0, xrt::bo& In1, xrt::bo& Out0, int app_id){
    unsigned int opcode = 3;
    LOG_VERBOSE(3, "Running kernel with app_id: " << app_id);
    auto run = this->hw_descs[app_id].kernel_desc->kernel(opcode, this->hw_descs[app_id].bo_instr, this->hw_descs[app_id].instr_size, In0, In1, Out0);
    ert_cmd_state r = run.wait();
    LOG_VERBOSE(3, "Kernel run finished with status: " << r);
    return r;
}

ert_cmd_state npu_app::run(xrt::bo& In0, xrt::bo& Out0, int app_id){
    unsigned int opcode = 3;
    LOG_VERBOSE(3, "Running kernel with app_id: " << app_id);
    auto run = this->hw_descs[app_id].kernel_desc->kernel(opcode, this->hw_descs[app_id].bo_instr, this->hw_descs[app_id].instr_size, In0, Out0);
    ert_cmd_state r = run.wait();
    LOG_VERBOSE(3, "Kernel run finished with status: " << r);
    return r;
}

xrt::run npu_app::create_run(xrt::bo& In0, xrt::bo& In1, xrt::bo& Out0, xrt::bo& Out1, int app_id){
    xrt::run run = xrt::run(this->hw_descs[app_id].kernel_desc->kernel);
    run.set_arg(0, 3);
    run.set_arg(1, this->hw_descs[app_id].bo_instr);
    run.set_arg(2, this->hw_descs[app_id].instr_size);
    run.set_arg(3, In0);
    run.set_arg(4, In1);
    run.set_arg(5, Out0);
    run.set_arg(6, Out1);
    return run;
}

xrt::run npu_app::create_run(xrt::bo& In0, xrt::bo& In1, xrt::bo& Out0, int app_id){
    xrt::run run = xrt::run(this->hw_descs[app_id].kernel_desc->kernel);
    run.set_arg(0, 3);
    run.set_arg(1, this->hw_descs[app_id].bo_instr);
    run.set_arg(2, this->hw_descs[app_id].instr_size);
    run.set_arg(3, In0);
    run.set_arg(4, In1);
    run.set_arg(5, Out0);
    return run;
}

xrt::run npu_app::create_run(xrt::bo& In0, xrt::bo& Out0, int app_id){
    xrt::run run = xrt::run(this->hw_descs[app_id].kernel_desc->kernel);
    run.set_arg(0, 3);
    run.set_arg(1, this->hw_descs[app_id].bo_instr);
    run.set_arg(2, this->hw_descs[app_id].instr_size);
    run.set_arg(3, In0);
    run.set_arg(4, Out0);
    return run;
}

xrt::runlist npu_app::create_runlist(int app_id){
    return xrt::runlist(this->hw_descs[app_id].kernel_desc->context);
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
        std::cout << "Instruction " << i << ": " << this->hw_descs[i].instr_name << std::endl;
    }
    std::cout << "Listing xclbins: (Total: " << this->kernel_descs.size() << ")" << std::endl;
    for (int i = 0; i < this->kernel_descs.size(); i++){
        std::cout << "Xclbin " << i << " at address: " <<  &this->kernel_descs[i].xclbin << std::endl;
    }
}

void npu_app::write_out_trace(char *traceOutPtr, size_t trace_size, std::string path) {
  std::ofstream fout(path);
  LOG_VERBOSE(1, "Writing out trace to: " << path);
  uint32_t *traceOut = (uint32_t *)traceOutPtr;
  for (int i = 0; i < trace_size / sizeof(traceOut[0]); i++) {
    fout << std::setfill('0') << std::setw(8) << std::hex << (int)traceOut[i];
    fout << std::endl;
  }
  fout.close();
  LOG_VERBOSE(1, "Trace written successfully!");
}

void npu_app::print_npu_info(){
    int fd = open("/dev/accel/accel0", O_RDWR);
    if (fd < 0) {
        perror("Failed to open amdgpu device");
        return;
    }
    amdxdna_drm_query_clock_metadata query_clock_metadata;
    amdxdna_drm_get_info get_info = {
        .param = DRM_AMDXDNA_QUERY_CLOCK_METADATA,
        .buffer_size = sizeof(amdxdna_drm_query_clock_metadata),
        .buffer = (unsigned long)&query_clock_metadata,
    };
    int ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
    if (ret < 0) {
        std::cout << "Error code: " << ret << std::endl;
        perror("Failed to get telemetry information");
        close(fd);
        return;
    }

    amdxdna_drm_query_aie_metadata query_aie_metadata;
    get_info.param = DRM_AMDXDNA_QUERY_AIE_METADATA;
    get_info.buffer_size = sizeof(amdxdna_drm_query_aie_metadata);
    get_info.buffer = (unsigned long)&query_aie_metadata;
    ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
    if (ret < 0) {
        std::cout << "Error code: " << ret << std::endl;
        perror("Failed to get telemetry information");
        close(fd);
        return;
    }

    close(fd);
    MSG_BONDLINE(40);
    MSG_BOX_LINE(40, "NPU version: " << query_aie_metadata.version.major << "." << query_aie_metadata.version.minor);
    MSG_BOX_LINE(40, "MP-NPU clock frequency: " << query_clock_metadata.mp_npu_clock.freq_mhz << " MHz");
    MSG_BOX_LINE(40, "H clock frequency: " << query_clock_metadata.h_clock.freq_mhz << " MHz");
    // What is the meaning of the column size?
    // std::cout << "NPU column size: " << query_aie_metadata.col_size << std::endl;
    MSG_BOX_LINE(40, "NPU column count: " << query_aie_metadata.cols);
    MSG_BOX_LINE(40, "NPU row count: " << query_aie_metadata.rows);
    MSG_BOX_LINE(40, "NPU core Info: ");
    MSG_BOX_LINE(40, "--Row count: " << query_aie_metadata.core.row_count);
    MSG_BOX_LINE(40, "--Row start: " << query_aie_metadata.core.row_start);
    MSG_BOX_LINE(40, "--DMA channel count: " << query_aie_metadata.core.dma_channel_count);
    MSG_BOX_LINE(40, "--Lock count: " << query_aie_metadata.core.lock_count);
    MSG_BOX_LINE(40, "--Event reg count: " << query_aie_metadata.core.event_reg_count);
    MSG_BOX_LINE(40, "NPU mem Info: ");
    MSG_BOX_LINE(40, "--Row count: " << query_aie_metadata.mem.row_count);
    MSG_BOX_LINE(40, "--Row start: " << query_aie_metadata.mem.row_start);
    MSG_BOX_LINE(40, "--DMA channel count: " << query_aie_metadata.mem.dma_channel_count);
    MSG_BOX_LINE(40, "--Lock count: " << query_aie_metadata.mem.lock_count);
    MSG_BOX_LINE(40, "--Event reg count: " << query_aie_metadata.mem.event_reg_count);
    MSG_BOX_LINE(40, "NPU shim Info: ");
    MSG_BOX_LINE(40, "--Row count: " << query_aie_metadata.shim.row_count);
    MSG_BOX_LINE(40, "--Row start: " << query_aie_metadata.shim.row_start);
    MSG_BOX_LINE(40, "--DMA channel count: " << query_aie_metadata.shim.dma_channel_count);
    MSG_BOX_LINE(40, "--Lock count: " << query_aie_metadata.shim.lock_count);
    MSG_BOX_LINE(40, "--Event reg count: " << query_aie_metadata.shim.event_reg_count);
    MSG_BONDLINE(40);
}

float npu_app::get_npu_power(bool print){
    // get the npu power consumption, unit is Watt
    int fd = open("/dev/accel/accel0", O_RDWR);
    if (fd < 0) {
        perror("Failed to open amdgpu device");
        return -1;
    }
    amdxdna_drm_query_sensor query_sensor;

    amdxdna_drm_get_info get_info = {
        .param = DRM_AMDXDNA_QUERY_SENSORS,
        .buffer_size = sizeof(amdxdna_drm_query_sensor),
        .buffer = (unsigned long)&query_sensor,
    };
    int ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
    if (ret < 0) {
        std::cout << "Error code: " << ret << std::endl;
        perror("Failed to get telemetry information");
        close(fd);
        return -1;
    }
    if (print){
        MSG_BOX(40, "NPU power: " << query_sensor.input << " " << query_sensor.units);
    }
    close(fd);
    return (float)query_sensor.input * pow(10, query_sensor.unitm);
}

void npu_app::interperate_bd(int app_id){
    // sync from the device to be consistent
    this->hw_descs[app_id].bo_instr.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    npu_sequence seq(this->hw_descs[app_id].bo_instr);
    seq.print_sequence();
    seq.to_npu();
}

std::vector<u_int64_t> npu_app::read_mem(uint32_t col, uint32_t row, uint32_t addr, uint32_t size){
    // read the register from the device
    std::vector<u_int64_t> regs(size);
    int fd = open("/dev/accel/accel0", O_RDWR);
    if (fd < 0) {
        perror("Failed to open npu device");
        return std::vector<u_int64_t>();
    }
    amdxdna_drm_aie_mem aie_mem = {
        .col = col,
        .row = row,
        .addr = addr,
        .size = size,
        .buf_p = (__u64)regs.data(),
    };

    amdxdna_drm_get_info get_info = {
        .param = DRM_AMDXDNA_READ_AIE_MEM,
        .buffer_size = sizeof(amdxdna_drm_aie_mem),
        .buffer = (unsigned long)&aie_mem,
    };
    int ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
    if (ret < 0) {
        std::cout << "Error code: " << ret << std::endl;
        perror("Failed to read memory");
        close(fd);
        return std::vector<u_int64_t>();
    }
    close(fd);
    return regs;
}

uint32_t npu_app::read_reg(uint32_t col, uint32_t row, uint32_t addr){
    // read the register from the device
    // __u32 val;
    // int fd = open("/dev/accel/accel0", O_RDWR);
    // if (fd < 0) {
    //     perror("Failed to open npu device");
    //     return 0;
    // }
    // amdxdna_drm_aie_reg aie_reg = {
    //     .col = col,
    //     .row = row,
    //     .addr = addr,
    //     .val = val,
    // };
    // amdxdna_drm_get_info get_info = {
    //     .param = DRM_AMDXDNA_READ_AIE_REG,
    //     .buffer_size = sizeof(amdxdna_drm_aie_reg),
    //     .buffer = (unsigned long)&aie_reg,
    // };
    // int ret = ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &get_info);
    // if (ret < 0) {
    //     std::cout << "Error code: " << ret << std::endl;
    //     perror("Failed to read register!");
    //     close(fd);
    //     return 0;
    // }
    // close(fd);
    // return val; // the value is already in the val variable
}

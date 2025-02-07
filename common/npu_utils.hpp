#ifndef __NPU_UTILS_HPP__
#define __NPU_UTILS_HPP__

#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <drm/drm.h>
#include <stdfloat>
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "amdxdna_accel.h"
#include "vector_view.hpp"
#include "debug_utils.hpp"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_ext.h"
#include "experimental/xrt_module.h"
#include "experimental/xrt_elf.h"
#include "xrt/xrt_graph.h"

#include "npu_instr_utils.hpp"
// Accelerator description
// There should be only one npu_app inside main.
typedef struct {
    std::string xclbin_name;
    std::string instr_name;
} accel_user_desc;

typedef struct {
    xrt::xclbin xclbin;
    xrt::kernel kernel;
    xrt::hw_context context;
} accel_kernel_desc;

typedef struct {
    std::string instr_name;
    accel_kernel_desc* kernel_desc;
    xrt::bo bo_instr;
    size_t instr_size;
} accel_hw_desc;


template <typename T>
struct AlignedAllocator {
    // Type definitions
    using value_type = T;

    // Allocate memory aligned to 4KB (4096 bytes)
    T* allocate(std::size_t n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 4096, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    // Deallocate memory
    void deallocate(T* ptr, std::size_t) noexcept {
        free(ptr);
    }
};

// npu_app
// There should be only one npu_app inside main.
// It handles all xclbins and instr_sequences.
// Each xclbin may have multiple instr_sequences.
// Each xclbin and instr_sequence has a unique id.
// Both id shall be provided to run an accelerator.
// Therefore, the xclbin_name between different accel_descriptions may overlap, but the instr_name is unique.
class npu_app{
private:
    std::vector<accel_kernel_desc> kernel_descs;
    std::vector<accel_hw_desc> hw_descs;
    std::vector<std::string> registered_xclbin_names;

    int kernel_desc_count;
    int hw_desc_count;

    // the only device instance
    xrt::device device;
public:
    npu_app(int max_xclbins = 1, int max_instrs = 1, unsigned int device_id = 0U);

    int register_accel_app(accel_user_desc& user_desc);
    ~npu_app();
    int _load_instr_sequence(accel_user_desc& user_desc, accel_hw_desc& hw_desc);
    int _load_xclbin(std::string xclbin_name);
    xrt::bo create_buffer(size_t size, int group_id, int app_id);
    template<typename T>
    vector<T> create_bo_vector(size_t size, int group_id, int app_id);

    ert_cmd_state run(xrt::bo& In0, xrt::bo& In1, xrt::bo& Out0, xrt::bo& Out1, int app_id = 0);
    ert_cmd_state run(xrt::bo& In0, xrt::bo& In1, xrt::bo& Out0, int app_id = 0);
    ert_cmd_state run(xrt::bo& In0, xrt::bo& Out0, int app_id = 0);

    xrt::run create_run(xrt::bo& In0, xrt::bo& In1, xrt::bo& Out0, xrt::bo& Out1, int app_id);
    xrt::run create_run(xrt::bo& In0, xrt::bo& In1, xrt::bo& Out0, int app_id);
    xrt::run create_run(xrt::bo& In0, xrt::bo& Out0, int app_id);

    xrt::runlist create_runlist(int app_id);
    
    void list_kernels();
    void write_out_trace(char *traceOutPtr, size_t trace_size, std::string path);
    void print_npu_info();
    float get_npu_power(bool print = true);

    void interperate_bd(int app_id);
    std::vector<u_int64_t> read_mem(uint32_t col, uint32_t row, uint32_t addr, uint32_t size);
    uint32_t read_reg(uint32_t col, uint32_t row, uint32_t addr);
};

#endif

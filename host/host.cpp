#include <bits/stdc++.h>
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

#include "typedef.hpp"
#include "npu_utils.hpp"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "common.hpp"

// Verification tolerance
// See "Note on Numerical Tolerances" in README.md
float abs_tol = matmul_common::get_abs_tol<C_DATATYPE>();
float rel_tol = matmul_common::get_rel_tol<C_DATATYPE>();

namespace po = boost::program_options;


void linear(vector<int32_t>& y, vector<int8_t>& w, vector<int8_t>& x);

int main(int argc, const char *argv[]) {
// Program arguments parsing
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    matmul_common::add_default_options(desc);

    matmul_common::parse_options(argc, argv, desc, vm);
    std::string dev = vm["D"].as<std::string>();
    int M = vm["M"].as<int>();
    int K = vm["K"].as<int>();
    int N = 1;
    int Y_VOLUME = M * 1;
    int W_VOLUME = M * K;
    int X_VOLUME = 1 * K;

    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    accel_user_desc accel_desc;
    accel_desc.xclbin_name = "build/xclbins/lm_head_i8.xclbin";
    accel_desc.instr_name = "build/insts/lm_head_i8.txt";
    std::cout << "Matrix size " << M << "x" << K << "x" << N << std::endl;

    npu_app npu_instance(1, 1, 0);
    int app_id = npu_instance.register_accel_app(accel_desc);

    vector<int8_t> w(W_VOLUME);
    vector<int8_t> x(X_VOLUME);
    vector<int32_t> y(Y_VOLUME);
    xrt::bo w_bo = npu_instance.create_buffer(W_VOLUME * sizeof(int8_t), 3, app_id);
    xrt::bo x_bo = npu_instance.create_buffer(X_VOLUME * sizeof(int8_t), 4, app_id);
    xrt::bo y_bo = npu_instance.create_buffer(Y_VOLUME * sizeof(int32_t), 5, app_id);
    
    y.remap(y_bo.map<int32_t*>(), Y_VOLUME);
    w.remap(w_bo.map<int8_t*>(), W_VOLUME);
    x.remap(x_bo.map<int8_t*>(), X_VOLUME);

    for (int i = 0; i < W_VOLUME; i++){
        int row = i / K;
        int col = i % K;
        if (row == col){
            w[i] = 1;
        } else {
            w[i] = 0;
        }
    }

    for (int i = 0; i < X_VOLUME; i++){
        if (i % 2 == 0){
            x[i] = 1;
        } else {
            x[i] = 0;
        }
        x[i] = i;
    }
 
    std::cout << "Calculate ref" << std::endl;
    vector<int32_t> y_ref(Y_VOLUME);    
    linear(y_ref, w, x);

    w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    x_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Running kernel" << std::endl;
    float npu_time = 0.0;
    for (int i = 0; i < 100; i++) {
	    auto start = std::chrono::high_resolution_clock::now();
	    npu_instance.run(w_bo, x_bo, y_bo, app_id);
	    auto stop = std::chrono::high_resolution_clock::now();
	    npu_time += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    }

    y_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);  
    std::cout << "Implement running kernel" << std::endl;

    bool pass = true;
    for (int i = 0; i < Y_VOLUME; i++){
        if (y[i] != y_ref[i]){
            std::cout << "y[" << i << "]: " << y[i] << " y_ref[" << i << "]: " << y_ref[i] << std::endl;
            pass = false;
            // break;
        }
    }
    for (int i = 0; i < 16; i++){
        std::cout << "y[" << i << "]: " << y[i] << " y_ref[" << i << "]: " << y_ref[i] << std::endl;
    }

    if (pass){
        std::cout << "PASS!" << std::endl;
    } else {
        std::cout << "FAIL!" << std::endl;
    }

    std::cout << "NPU time: " << npu_time << "us." << std::endl;

    std::cout << "NPU gflops: " << 2.0 * float(M) * float(K) * float(N) / (1000 * npu_time / 100.0) << std::endl;

    return 0;
}

void linear(vector<int32_t>& y, vector<int8_t>& w, vector<int8_t>& x){
    int in_features = x.size();
    int out_features = y.size();
    assert((in_features * out_features) == w.size());
    int v = 0;
    for (int row = 0; row < out_features; row++){
        int32_t sum = 0;
        for (int col = 0; col < in_features; col++){
            sum = sum + int32_t(w[v++] * x[col]);
        }
        y[row] = sum;
    }
}

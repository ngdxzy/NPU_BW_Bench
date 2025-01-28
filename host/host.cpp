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
#include "debug_utils.hpp"

#include "typedef.hpp"
#include "npu_utils.hpp"
#include "vm_args.hpp"
#include "utils.hpp"
namespace po = boost::program_options;


void linear(vector<dtype_out>& y, vector<dtype_in>& w, vector<dtype_in>& x);

int main(int argc, const char *argv[]) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options
    desc.add_options()("M,m", po::value<int>()->default_value(2048), "M");
    desc.add_options()("K,k", po::value<int>()->default_value(2048), "K");

    arg_utils::parse_options(argc, argv, desc, vm);
    
    // User logic
    int M = vm["M"].as<int>();
    int K = vm["K"].as<int>();
    int N = 1;
    int Y_VOLUME = M * 1;
    int W_VOLUME = M * K;
    int X_VOLUME = 1 * K;

    // NPU instance
    npu_app npu_instance(1, 1, 0);
    if (VERBOSE >= 1){
        npu_instance.get_npu_power(true);
        npu_instance.print_npu_info();
    }

    accel_user_desc accel_desc = {
        .xclbin_name = "build/xclbins/mvm_i8.xclbin",
        .instr_name = "build/insts/mvm_i8.txt",
    };
    
    header_print("info", "Matrix size " << M << "x" << K << "x" << N);

    int app_id = npu_instance.register_accel_app(accel_desc);

    vector w = npu_instance.create_bo_vector<dtype_in>(W_VOLUME, 3, app_id);
    vector x = npu_instance.create_bo_vector<dtype_in>(X_VOLUME, 4, app_id);
    vector y = npu_instance.create_bo_vector<dtype_out>(Y_VOLUME, 5, app_id);

    for (int i = 0; i < W_VOLUME; i++){
        w[i] = utils::getRandInt(-10, 10);
    }

    for (int i = 0; i < X_VOLUME; i++){
        x[i] = utils::getRandInt(-10, 10);
    }
 
    header_print("info", "Calculate ref");
    vector<dtype_out> y_ref(Y_VOLUME);    
    linear(y_ref, w, x);

    w.sync_to_device();
    x.sync_to_device();

    header_print("info", "Running kernel");
    float npu_time = 0.0;
    for (int i = 0; i < 100; i++) {
	    time_utils::time_point start = time_utils::now();
	    npu_instance.run(w.bo(), x.bo(), y.bo(), app_id);
	    time_utils::time_point stop = time_utils::now();
	    npu_time += time_utils::duration_us(start, stop).first;
    }

    y.sync_from_device();  
    header_print("info", "Finished running kernel");

    bool pass = true;
    if (utils::compare_vectors(y, y_ref) > 0){
        pass = false;
    }

    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }

    utils::print_npu_profile(npu_time, 2.0 * float(M) * float(K) * float(N));
    return 0;
}

void linear(vector<dtype_out>& y, vector<dtype_in>& w, vector<dtype_in>& x){
    int in_features = x.size();
    int out_features = y.size();
    assert((in_features * out_features) == w.size());
    int v = 0;
    for (int row = 0; row < out_features; row++){
        dtype_out sum = 0;
        for (int col = 0; col < in_features; col++){
            sum = sum + dtype_out(w[v++] * x[col]);
        }
        y[row] = sum;
    }
}

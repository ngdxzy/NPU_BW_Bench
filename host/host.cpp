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
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_queue.h"

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
    desc.add_options()("M,m", po::value<int>()->default_value(128 * 16), "M");
    desc.add_options()("K,k", po::value<int>()->default_value(128 * 4), "K");

    arg_utils::parse_options(argc, argv, desc, vm);
    
    // User logic
    int M = vm["M"].as<int>();
    int K = vm["K"].as<int>();
    int N = 1;
    int Y_VOLUME = M * 1;
    int W_VOLUME = M * K;
    int X_VOLUME = 1 * K;

    // NPU instance
    npu_app npu_instance(1, 2, 0);
    if (VERBOSE >= 1){
        npu_instance.get_npu_power(true);
        npu_instance.print_npu_info();
    }

    accel_user_desc accel_desc_0 = {
        .xclbin_name = "build/xclbins/mvm_i8.xclbin",
        .instr_name = "build/insts/mvm_i8.txt",
    };
    accel_user_desc accel_desc_1 = {
        .xclbin_name = "build/xclbins/mvm_i8.xclbin",
        .instr_name = "build/insts/mvm_i8.txt.redundant",
    };

    
    header_print("info", "Matrix size " << M << "x" << K << "x" << N);

    int app_id_0 = npu_instance.register_accel_app(accel_desc_0);
    int app_id_1 = npu_instance.register_accel_app(accel_desc_1);

    npu_instance.print_npu_info();

    npu_instance.list_kernels();

    npu_instance.interperate_bd(0);
    // npu_instance.interperate_bd(1); They are the same


    vector w_0 = npu_instance.create_bo_vector<dtype_in>(W_VOLUME, 3, app_id_0);
    vector x_0 = npu_instance.create_bo_vector<dtype_in>(X_VOLUME, 4, app_id_0);
    vector y_0 = npu_instance.create_bo_vector<dtype_out>(Y_VOLUME, 5, app_id_0);

    vector w_1 = npu_instance.create_bo_vector<dtype_in>(W_VOLUME, 3, app_id_1);
    vector x_1 = npu_instance.create_bo_vector<dtype_in>(X_VOLUME, 4, app_id_1);
    vector y_1 = npu_instance.create_bo_vector<dtype_out>(Y_VOLUME, 5, app_id_1);

    for (int i = 0; i < W_VOLUME; i++){
        w_0[i] = utils::getRandInt(-10, 10);
        w_1[i] = utils::getRandInt(-10, 10);
    }

    for (int i = 0; i < X_VOLUME; i++){
        x_0[i] = utils::getRandInt(-10, 10);
        x_1[i] = utils::getRandInt(-10, 10);
    }
 
    header_print("info", "Calculate reference for " << M << "x" << K << "x" << N);
    vector<dtype_out> y_ref_0(Y_VOLUME);    
    vector<dtype_out> y_ref_1(Y_VOLUME);    
    linear(y_ref_0, w_0, x_0);
    linear(y_ref_1, w_1, x_1);

    w_0.sync_to_device();
    w_1.sync_to_device();
    x_0.sync_to_device();
    x_1.sync_to_device();

    header_print("info", "Running kernel with bare call.");
    time_utils::time_with_unit npu_time = {0.0, "us"};
    for (int i = 0; i < 1000; i++) {
	    time_utils::time_point start = time_utils::now();
	    npu_instance.run(w_0.bo(), x_0.bo(), y_0.bo(), app_id_0);
	    npu_instance.run(w_1.bo(), x_1.bo(), y_1.bo(), app_id_1);
	    time_utils::time_point stop = time_utils::now();
	    npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= 1000 * 2;
    MSG_BONDLINE(40);
    MSG_BOX_LINE(40, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(40);

    y_0.sync_from_device();    
    y_1.sync_from_device();
    header_print("info", "Finished running kernel");

    bool pass = true;
    if (utils::compare_vectors(y_0, y_ref_0) > 0){
        pass = false;
    }
    if (utils::compare_vectors(y_1, y_ref_1) > 0){
        pass = false;
    }
    // run with runlist
    xrt::runlist runlist = npu_instance.create_runlist(app_id_0);
    y_0.memset(0);
    y_1.memset(0);
    for (int i = 0; i < 1000; i++){
        xrt::run run_0 = npu_instance.create_run(w_0.bo(), x_0.bo(), y_0.bo(), app_id_0);
        xrt::run run_1 = npu_instance.create_run(w_1.bo(), x_1.bo(), y_1.bo(), app_id_1);
        runlist.add(run_0);
        runlist.add(run_1);
    }
    
    npu_time.first = 0;

    {
        time_utils::time_point start = time_utils::now();
        runlist.execute();
        runlist.wait();
        time_utils::time_point stop = time_utils::now();
        npu_time.first += time_utils::duration_us(start, stop).first;
    }
    npu_time.first /= 1000 * 2;
    MSG_BONDLINE(40);
    MSG_BOX_LINE(40, "NPU time with runlist: " << npu_time.first << " us");
    MSG_BONDLINE(40);
    y_0.sync_from_device();    
    y_1.sync_from_device();

    if (utils::compare_vectors(y_0, y_ref_0) > 0){
        pass = false;
    }
    if (utils::compare_vectors(y_1, y_ref_1) > 0){
        pass = false;
    }
    

    // run with queue   
    // xrt::queue q = xrt::queue();
    // npu_time.first = 0;
    // std::vector<std::shared_future<void>> events;
    // {
    //     time_utils::time_point start = time_utils::now();
    //     for (int i = 0; i < 1000; i++){
    //         xrt::run run = npu_instance.create_run(w.bo(), x.bo(), y.bo(), app_id);
    //         auto event = q.enqueue([&]() {
    //             run.start();
    //             run.wait();  // Wait for completion
    //         });
    //         events.push_back(event);
    //     }
    //     for (auto event : events){
    //         event.wait();
    //     }
    //     time_utils::time_point stop = time_utils::now();
    //     npu_time.first += time_utils::duration_us(start, stop).first;
    //     npu_time.first /= 1000;
    // }
    // MSG_BONDLINE(40);
    // MSG_BOX_LINE(40, "NPU time with queue: " << npu_time.first << " us");
    // MSG_BONDLINE(40);

    if (pass){
        header_print("info", "PASSED ");
    } else {
        header_print("info", "FAILED!");
    }

    // utils::print_npu_profile(npu_time, 2.0 * float(M) * float(K) * float(N), 1000);
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

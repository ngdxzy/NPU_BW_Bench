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

int main(int argc, const char *argv[]) {
    // Fix the seed to ensure reproducibility in CI.
    srand(0);
    // Program arguments parsing
    po::options_description desc("Allowed options");
    po::variables_map vm;
    arg_utils::add_default_options(desc);

    // Add custom options
    desc.add_options()("I,i", po::value<int>()->default_value(0), "Iterations");
    desc.add_options()("T,t", po::value<int>()->default_value(4), "Trace length");

    arg_utils::parse_options(argc, argv, desc, vm);
    
    // User logic
    int Iterations = vm["I"].as<int>();
    int TraceLength = vm["T"].as<int>();

    // NPU instance
    npu_app npu_instance(1, 1, 0);
    if (VERBOSE >= 1){
        npu_instance.get_npu_power(true);
        npu_instance.print_npu_info();
    }

    accel_user_desc accel_desc = {
        .xclbin_name = "build/xclbins/bwbench.xclbin",
        .instr_name = "build/insts/bwbench.txt",
    };

    int app_id = npu_instance.register_accel_app(accel_desc);

    npu_instance.print_npu_info();
    npu_instance.list_kernels();
    // npu_instance.interperate_bd(0);

    const int use_cols = 4;
    const int token_rate = 32;
    const int burst_size = 1024;
    const int rounds = 4;
    const int A_size = use_cols * burst_size * token_rate * 2;
    const int C_size = use_cols * burst_size;
    const int T_size = TraceLength / 4;

    
    buffer<uint32_t> A = npu_instance.create_bo_buffer<uint32_t>(A_size, 3, app_id);
    buffer<uint32_t> C = npu_instance.create_bo_buffer<uint32_t>(C_size, 4, app_id);
    buffer<uint32_t> T = npu_instance.create_bo_buffer<uint32_t>(T_size, 5, app_id);

    A.memset(0);
    C.memset(0);
    T.memset(0);

    A.sync_to_device();
    C.sync_to_device();
    T.sync_to_device();

	
    header_print("info", "Running runtime test.");
    header_print("info", "Running kernel with bare call.");
    time_utils::time_with_unit npu_time = {0.0, "us"};
	
    auto run = npu_instance.create_run(A.bo(), C.bo(), T.bo(), app_id);
    time_utils::time_point start = time_utils::now();
    run.start();
    run.wait();
    time_utils::time_point stop = time_utils::now();
    npu_time.first += time_utils::duration_us(start, stop).first;
    header_print("info", "Finished running kernel");
    MSG_BONDLINE(40);
    MSG_BOX_LINE(40, "NPU time with bare call: " << npu_time.first << " us");
    MSG_BONDLINE(40);


    // run with runlist
    if (Iterations > 0){
        header_print("info", "Benchmarking!");
        const int statistic_minimal = 512;
        int current_iter = 0;
        int total_iter = std::max(Iterations, statistic_minimal);
        npu_time.first = 0;
        while (current_iter < total_iter){
            auto runlist = npu_instance.create_runlist(app_id);
            for (int i = 0; i < Iterations; i++){
                runlist.add(npu_instance.create_run(A.bo(), C.bo(), T.bo(), app_id));
            }
            time_utils::time_point start = time_utils::now();
            runlist.execute();
            runlist.wait();
            time_utils::time_point stop = time_utils::now();
            npu_time.first += time_utils::duration_us(start, stop).first;
            current_iter += Iterations;
            utils::print_progress_bar(std::cout, current_iter / (float)total_iter, 40);
        }
        std::cout << std::endl;
        npu_time.first /= current_iter;
        float achieved_bandwidth = rounds * token_rate * burst_size * 2 * use_cols * 4; // total bytes read from DDR
        achieved_bandwidth = achieved_bandwidth / (npu_time.first / 1e6); // bytes per second
        achieved_bandwidth = achieved_bandwidth / 1024 / 1024 / 1024; // GiB/s
        MSG_BONDLINE(40);
        MSG_BOX_LINE(40, "NPU time with runlist: " << npu_time.first << " us");
        MSG_BOX_LINE(40, "Achieved bandwidth   : " << achieved_bandwidth << " GiB/s");
        MSG_BONDLINE(40);
        C.sync_from_device();    
        T.sync_from_device();
    }

    return 0;
}
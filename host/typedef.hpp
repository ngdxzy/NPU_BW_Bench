#ifndef __TYPEDEF_H__
#define __TYPEDEF_H__
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

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "vector_view.hpp"

#ifndef DATATYPES_USING_DEFINED

#define DATATYPES_USING_DEFINED

#ifndef DTYPE_IN
#define DTYPE_IN std::bfloat16_t
#endif

#ifndef DTYPE_OUT
#define DTYPE_OUT std::bfloat16_t
#endif

#ifndef DTYPE_ACC
#define DTYPE_ACC std::bfloat16_t

#endif

using A_DATATYPE = DTYPE_IN;
using B_DATATYPE = DTYPE_IN;
using C_DATATYPE = DTYPE_OUT;
using ACC_DATATYPE = DTYPE_OUT;

#endif

typedef A_DATATYPE dtype;
typedef float accdtype;
typedef vector<dtype> vdtype;

#endif

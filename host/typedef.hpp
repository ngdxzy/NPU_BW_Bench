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

#endif

typedef vector<DTYPE_IN> vdtype;

#endif

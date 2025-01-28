#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

#include "zero.h"
#include "vvm.h"
#include "passThrough_aie.h"

extern "C" {

#ifndef DIM_dk
#define DIM_dk 256
#endif

#ifndef DIM_dv
#define DIM_dv 16
#endif

void attentionmap(
        bfloat16 *restrict akq,
        bfloat16 *restrict vs,
        bfloat16 *restrict sout
){
    vecvec_vectorized<bfloat16, accfloat, DIM_dk, DIM_dv>(akq, vs, sout);
}


void zero_v(
    bfloat16 *restrict c
) {
    zero_vectorized<bfloat16, DIM_dv>(c);
}
} // extern "C"

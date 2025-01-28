#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

#include "zero.h"
#include "passThrough_aie.h"

template <typename T, typename T_out, typename T_acc, unsigned m, unsigned k, unsigned r>
void matvec_vectorized(T *__restrict a, T *__restrict b, T_out *__restrict c) {
    static_assert(m % r == 0 && k % 2 == 0);
    static_assert(std::is_same<T, bfloat16>::value || std::is_same<T, int16_t>::value);

    event0();
    T *__restrict a_ptr = a;
    T *__restrict b_ptr = b;
    T_out *__restrict c_ptr = c; // reset to the first row of C output on

    for (int row = 0; row < m; row += r)
    chess_prepare_for_pipelining chess_loop_range(2, ){
        aie::accum<T_acc, r> c_acc_in;
        c_acc_in.from_vector(aie::load_v<r>(c_ptr));

        a_ptr = a + 2 * row;
        b_ptr = b;
        for (int col = 0; col < k; col += 8) 
        chess_flatten_loop {
            aie::vector<T, 8> b_vec = aie::load_v<8>(b_ptr);

            const aie::vector<T, 2 * r> a_vec_0 = aie::load_v<2 * r>(a_ptr);
            const aie::vector<T, 2 * r> a_vec_1 = aie::load_v<2 * r>(a_ptr + 2 * m);
            const aie::vector<T, 2 * r> a_vec_2 = aie::load_v<2 * r>(a_ptr + 4 * m);
            const aie::vector<T, 2 * r> a_vec_3 = aie::load_v<2 * r>(a_ptr + 6 * m);

            const aie::vector<T, r> a_vec_0_0 = aie::filter_even(a_vec_0, 1);
            const aie::vector<T, r> a_vec_0_1 = aie::filter_odd(a_vec_0, 1);
            const aie::vector<T, r> a_vec_1_0 = aie::filter_even(a_vec_1, 1);
            const aie::vector<T, r> a_vec_1_1 = aie::filter_odd(a_vec_1, 1);
            const aie::vector<T, r> a_vec_2_0 = aie::filter_even(a_vec_2, 1);
            const aie::vector<T, r> a_vec_2_1 = aie::filter_odd(a_vec_2, 1);
            const aie::vector<T, r> a_vec_3_0 = aie::filter_even(a_vec_3, 1);
            const aie::vector<T, r> a_vec_3_1 = aie::filter_odd(a_vec_3, 1);

            c_acc_in = aie::accumulate<r>(c_acc_in, b_vec, 0, a_vec_0_0, a_vec_0_1, a_vec_1_0, a_vec_1_1, a_vec_2_0, a_vec_2_1, a_vec_3_0, a_vec_3_1);

            a_ptr += 8 * m; // Move to next 8 columns of A.
            b_ptr += 8;     // Move to next s (==8) rows of b.
        }
        aie::store_v(c_ptr, c_acc_in.template to_vector<T_out>());
        c_ptr += r;     // Move to next r rows of the same columns in A.
    }
    event1();
}


extern "C" {

#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

void mv(
        bfloat16 *restrict a,
        bfloat16 *restrict b,
        float *restrict c
){
    matvec_vectorized<bfloat16, float, accfloat, DIM_M, DIM_K, 16>(a, b, c);
}

void zero_m(
    float *restrict c
) {
    zero_vectorized<float, DIM_M>(c);
}

void passThrough(
    bfloat16 *restrict Out,
    float *restrict In
    ){
    passThrough_aie<float, bfloat16>(In, Out, DIM_M);
}
} // extern "C"

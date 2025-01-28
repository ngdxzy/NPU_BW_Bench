#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

#include "zero.h"


template <typename T, typename T_out, typename T_acc, unsigned m, unsigned k, unsigned r>
void matvec_vectorized_int8(T *__restrict a, T *__restrict b, T_out *__restrict c) {
    static_assert(m % r == 0 && k % 2 == 0);
    static_assert(std::is_same<T, bfloat16>::value || std::is_same<T, int8_t>::value);

    // for (int i = 0; i < m; i++){
    //     // c[i] = 0;
    //     for (int j = 0; j < k; j++){
    //         c[i] += a[i * k + j] * b[j];
    //     }
    // }
    // return;
    event0();
    T *__restrict a_ptr = a;
    T *__restrict b_ptr = b;
    T_out *__restrict c_ptr = c; // reset to the first row of C output on

    for (int row = 0; row < m; row += r)
    chess_prepare_for_pipelining chess_loop_range(2, ){
        aie::accum<T_acc, r> c_acc_in;
        c_acc_in.from_vector(aie::load_v<r>(c_ptr));

        a_ptr = a + 4 * row;
        b_ptr = b;
        for (int col = 0; col < k; col += 16) 
        chess_flatten_loop {
            aie::vector<T, 16> b_vec = aie::load_v<16>(b_ptr);

            const aie::vector<T, 4 * r> a_vec_0 = aie::load_v<4 * r>(a_ptr);
            const aie::vector<T, 4 * r> a_vec_1 = aie::load_v<4 * r>(a_ptr + 4 * m);
            const aie::vector<T, 4 * r> a_vec_2 = aie::load_v<4 * r>(a_ptr + 8 * m);
            const aie::vector<T, 4 * r> a_vec_3 = aie::load_v<4 * r>(a_ptr + 12 * m);

            const aie::vector<T, 2 * r> a_vec_00 = aie::filter_even(a_vec_0, 2);
            const aie::vector<T, 2 * r> a_vec_01 = aie::filter_odd( a_vec_0, 2);
            const aie::vector<T, 2 * r> a_vec_10 = aie::filter_even(a_vec_1, 2);
            const aie::vector<T, 2 * r> a_vec_11 = aie::filter_odd( a_vec_1, 2);
            const aie::vector<T, 2 * r> a_vec_20 = aie::filter_even(a_vec_2, 2);
            const aie::vector<T, 2 * r> a_vec_21 = aie::filter_odd( a_vec_2, 2);
            const aie::vector<T, 2 * r> a_vec_30 = aie::filter_even(a_vec_3, 2);
            const aie::vector<T, 2 * r> a_vec_31 = aie::filter_odd( a_vec_3, 2);
          

            const aie::vector<T, r> a_vec_0_0 = aie::filter_even(a_vec_00, 1);
            const aie::vector<T, r> a_vec_0_1 = aie::filter_odd(a_vec_00, 1);
            const aie::vector<T, r> a_vec_1_0 = aie::filter_even(a_vec_01, 1);
            const aie::vector<T, r> a_vec_1_1 = aie::filter_odd(a_vec_01, 1);
            const aie::vector<T, r> a_vec_2_0 = aie::filter_even(a_vec_10, 1);
            const aie::vector<T, r> a_vec_2_1 = aie::filter_odd(a_vec_10, 1);
            const aie::vector<T, r> a_vec_3_0 = aie::filter_even(a_vec_11, 1);
            const aie::vector<T, r> a_vec_3_1 = aie::filter_odd(a_vec_11, 1);
            const aie::vector<T, r> a_vec_4_0 = aie::filter_even(a_vec_20, 1);
            const aie::vector<T, r> a_vec_4_1 = aie::filter_odd(a_vec_20, 1);
            const aie::vector<T, r> a_vec_5_0 = aie::filter_even(a_vec_21, 1);
            const aie::vector<T, r> a_vec_5_1 = aie::filter_odd(a_vec_21, 1);
            const aie::vector<T, r> a_vec_6_0 = aie::filter_even(a_vec_30, 1);
            const aie::vector<T, r> a_vec_6_1 = aie::filter_odd(a_vec_30, 1);
            const aie::vector<T, r> a_vec_7_0 = aie::filter_even(a_vec_31, 1);
            const aie::vector<T, r> a_vec_7_1 = aie::filter_odd(a_vec_31, 1);
            // const aie::vector<T, r> a_vec_8_0 = aie::filter_even(a_vec_8, 1);
            // const aie::vector<T, r> a_vec_8_1 = aie::filter_odd(a_vec_8, 1);
            // const aie::vector<T, r> a_vec_9_0 = aie::filter_even(a_vec_9, 1);
            // const aie::vector<T, r> a_vec_9_1 = aie::filter_odd(a_vec_9, 1);
            // const aie::vector<T, r> a_vec_10_0 = aie::filter_even(a_vec_10, 1);
            // const aie::vector<T, r> a_vec_10_1 = aie::filter_odd(a_vec_10, 1);
            // const aie::vector<T, r> a_vec_11_0 = aie::filter_even(a_vec_11, 1);
            // const aie::vector<T, r> a_vec_11_1 = aie::filter_odd(a_vec_11, 1);
            // const aie::vector<T, r> a_vec_12_0 = aie::filter_even(a_vec_12, 1);
            // const aie::vector<T, r> a_vec_12_1 = aie::filter_odd(a_vec_12, 1);
            // const aie::vector<T, r> a_vec_13_0 = aie::filter_even(a_vec_13, 1);
            // const aie::vector<T, r> a_vec_13_1 = aie::filter_odd(a_vec_13, 1);
            // const aie::vector<T, r> a_vec_14_0 = aie::filter_even(a_vec_14, 1);
            // const aie::vector<T, r> a_vec_14_1 = aie::filter_odd(a_vec_14, 1);
            // const aie::vector<T, r> a_vec_15_0 = aie::filter_even(a_vec_15, 1);
            // const aie::vector<T, r> a_vec_15_1 = aie::filter_odd(a_vec_15, 1);

            c_acc_in = aie::accumulate<r>(c_acc_in, b_vec, 0, 
                a_vec_0_0, a_vec_0_1, 
                a_vec_1_0, a_vec_1_1, 
                a_vec_2_0, a_vec_2_1, 
                a_vec_3_0, a_vec_3_1, 
                a_vec_4_0, a_vec_4_1, 
                a_vec_5_0, a_vec_5_1, 
                a_vec_6_0, a_vec_6_1, 
                a_vec_7_0, a_vec_7_1
                // a_vec_8_0, a_vec_8_1, 
                // a_vec_9_0, a_vec_9_1, 
                // a_vec_10_0, a_vec_10_1, 
                // a_vec_11_0, a_vec_11_1, 
                // a_vec_12_0, a_vec_12_1, 
                // a_vec_13_0, a_vec_13_1, 
                // a_vec_14_0, a_vec_14_1, 
                // a_vec_15_0, a_vec_15_1
            );

            a_ptr += 16 * m; // Move to next 8 columns of A.
            b_ptr += 16;     // Move to next s (==8) rows of b.
        }
        aie::store_v(c_ptr, c_acc_in.template to_vector<T_out>());
        c_ptr += r;     // Move to next r rows of the same columns in A.
    }
    event1();
}

extern "C" {

#ifndef DIM_M_I8
#define DIM_M_I8 128
#endif

#ifndef DIM_K_I8
#define DIM_K_I8 128
#endif

void mv_int8(
        int8 *restrict a,
        int8 *restrict b,
        int32 *restrict c
){
    matvec_vectorized_int8<int8, int32, acc32, DIM_M_I8, DIM_K_I8, 16>(a, b, c);
}

void zero_m_int8(
    int32 *restrict c
) {
    zero_vectorized<int32, DIM_M_I8>(c);
}

} // extern "C"

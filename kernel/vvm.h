#ifndef __VV_H__
#define __VV_H__

#include <stdint.h>

#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

template <typename T, typename T_acc, unsigned dk, unsigned dv>
void vecvec_vectorized(T *__restrict akq, T *__restrict vs, T *__restrict os_out) {

    event0();
    T *__restrict a_ptr = akq;
    T *__restrict k_ptr = akq + dk;
    T *__restrict q_ptr = akq + dk + dk;
    T *__restrict v_ptr = vs;
    T *__restrict s_ptr = vs + dv;
    T *__restrict o_ptr = os_out;
    T *__restrict so_ptr = os_out + dv; 

    const int vector_fac = 16;
    const int quad_dk = dk / 4;
    const int quad_dv = dv / 4;
    const int quad_s = dk * dv / 4;

    for (int row = 0; row < dv; row += vector_fac){
        aie::accum<T_acc, vector_fac> o_acc;
        o_acc.from_vector(aie::load_v<vector_fac>(o_ptr));
        
        so_ptr = os_out + dv + row;
        s_ptr = vs + dv + row;
        v_ptr = vs + row;

        for (int col = 0; col < dk / 4; col += 1)
        chess_prepare_for_pipelining chess_loop_range(dk / 4, ) {
        //chess_flatten_loop{
            // load s_{t-1}
            aie::vector<T, vector_fac> s_vec_1 = aie::load_v<vector_fac>(s_ptr);
            s_ptr += quad_s;
            aie::vector<T, vector_fac> s_vec_2 = aie::load_v<vector_fac>(s_ptr);
            s_ptr += quad_s;
            aie::vector<T, vector_fac> s_vec_3 = aie::load_v<vector_fac>(s_ptr);
            s_ptr += quad_s;
            aie::vector<T, vector_fac> s_vec_4 = aie::load_v<vector_fac>(s_ptr);
            s_ptr -= quad_s * 3;

            // alpha * s_{t-1}
            aie::accum<T_acc, vector_fac> s_acc_1 = aie::mul(a_ptr[col], s_vec_1);
            aie::accum<T_acc, vector_fac> s_acc_2 = aie::mul(a_ptr[col + quad_dk], s_vec_2);
            aie::accum<T_acc, vector_fac> s_acc_3 = aie::mul(a_ptr[col + 2 * quad_dk], s_vec_3);
            aie::accum<T_acc, vector_fac> s_acc_4 = aie::mul(a_ptr[col + 3 * quad_dk], s_vec_4);
            //load v
            aie::vector<T, vector_fac> v_vec = aie::load_v<vector_fac>(v_ptr);
            // alpha * s_{t-1} + K^T V
            s_acc_1 = aie::mac(s_acc_1, k_ptr[col], v_vec);
            s_acc_2 = aie::mac(s_acc_2, k_ptr[col + quad_dk], v_vec);
            s_acc_3 = aie::mac(s_acc_3, k_ptr[col + 2 * quad_dk], v_vec);
            s_acc_4 = aie::mac(s_acc_4, k_ptr[col + 3 * quad_dk], v_vec);

            // o
            T q_1 = T((float)q_ptr[col] * 0.0625);
            T q_2 = T((float)q_ptr[col + quad_dk] * 0.0625);
            T q_3 = T((float)q_ptr[col + 2 * quad_dk] * 0.0625);
            T q_4 = T((float)q_ptr[col + 3 * quad_dk] * 0.0625);
            o_acc = aie::mac(o_acc, q_1, s_acc_1.template to_vector<T>());
            o_acc = aie::mac(o_acc, q_2, s_acc_2.template to_vector<T>());
            o_acc = aie::mac(o_acc, q_3, s_acc_3.template to_vector<T>());
            o_acc = aie::mac(o_acc, q_4, s_acc_4.template to_vector<T>());

            aie::store_v(so_ptr, s_acc_1.template to_vector<T>());
            so_ptr += quad_s;
            aie::store_v(so_ptr, s_acc_2.template to_vector<T>());
            so_ptr += quad_s;
            aie::store_v(so_ptr, s_acc_3.template to_vector<T>());
            so_ptr += quad_s;
            aie::store_v(so_ptr, s_acc_4.template to_vector<T>());
            so_ptr -= quad_s * 3;

            so_ptr += dv;
            s_ptr += dv;
        }
        aie::store_v(o_ptr, o_acc.template to_vector<T>());
        o_ptr += vector_fac;
    }
    // for (int col = 0; col < dk; col ++){
    // chess_prepare_for_pipelining chess_loop_range(dk, ) {
    //     for (int row = 0; row < dv; row += vector_fac){
    //     //chess_flatten_loop{
    //         // load s_{t-1}
    //         aie::vector<T, vector_fac> s_vec = aie::load_v<vector_fac>(s_ptr);
    //         // alpha * s_{t-1}
    //         aie::accum<T_acc, vector_fac> s_acc = aie::mul(a_ptr[col], s_vec);
    //         //load v
    //         aie::vector<T, vector_fac> v_vec = aie::load_v<vector_fac>(v_ptr);
    //         // alpha * s_{t-1} + K^T V
    //         aie::accum<T_acc, vector_fac> sout_acc = aie::mac(s_acc, k_ptr[col], v_vec);

    //         // o
    //         aie::accum<T_acc, vector_fac> o_acc;
    //         o_acc.from_vector(aie::load_v<vector_fac>(o_ptr));
    //         aie::accum<T_acc, vector_fac> o_acc = aie::mac(o_acc, q_ptr[col], sout_acc.template to_vector<T>());

    //         aie::store_v(so_ptr, sout_acc.template to_vector<T>());
    //         aie::store_v(o_ptr, o_acc.template to_vector<T>());
    //         o_ptr += vector_fac;
    //         so_ptr += vector_fac;
    //         s_ptr += vector_fac;
    //         v_ptr += vector_fac;
    //     }
    //     o_ptr -= dv;
    //     v_ptr -= dv;
    // }
    event1();
}

#endif

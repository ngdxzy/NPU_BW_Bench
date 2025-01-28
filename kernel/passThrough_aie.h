#ifndef __PASSTHROUGH_AIE_H__
#define __PASSTHROUGH_AIE_H__

template <typename T_in, typename T_out>
void passThrough_aie(T_in *restrict in0, T_out *restrict out, const int N) {
    event0();
    const int vec_factor = 16;

    aie::vector<T_out, vec_factor> Out;

    const int F = N / vec_factor;
    for (int i = 0; i < F; i++)
        chess_flatten_loop chess_loop_range(2, ) { 
            aie::accum<accfloat, vec_factor> c_acc_in;
            c_acc_in.from_vector(aie::load_v<vec_factor>(in0));
            aie::store_v(out, c_acc_in.template to_vector<T_out>());
            in0 += vec_factor;
            out += vec_factor;
        }
    event1();
}
#endif

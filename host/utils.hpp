#ifndef __UTILS_H__
#define __UTILS_H__
#include "common.hpp"
#include "typedef.hpp"


float getRand(){
    return (float)(rand() % 1000) / 500.0 - 1;
}

template<typename T>
void transpose(T* A, T* B, int Ma, int Na) 
{ 
    // A: M N
    // B: N M
    int n, m; 
    for (n = 0; n < Na; n++) 
        for (m = 0; m < Ma; m++) 
           // B[n][m] = A[m][n]; 
           B[n * Ma + m] = A[m * Na + n];
} 


void init_A(A_DATATYPE* AVec, bool random){
    for (int i = 0; i < M * K; i++) {
        AVec[i] =  20 * getRand() / K; 
        if (random){
            continue;
        }

        int m = i / K;
        m =  m %  K;
        int k = i % K;

        if (m == k){
            AVec[i] = 1;
        }
        else{
            AVec[i] = 0.000;
        }
    }
}

void init_B(B_DATATYPE* BVec, bool random){
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            if (random){
                BVec[k * N + n] = 1 * getRand() ;
            }
            else{
                BVec[k * N + n] = n + 1 + (k + 1) / 10.0;
            }
        }
    }
}

void load_W(C_DATATYPE* bufA, A_DATATYPE* AVec){
// Hardware read A for each CT, each CT read M_PER_CT * K_TILE
    memcpy(bufA, AVec, M * K * sizeof(A_DATATYPE));
    return;
    for (int mt = 0; mt < M / M_PER_CT / CT_ROW; mt++){
        for (int kt = 0; kt < K / K_TILE; kt++){
            for (int ct = 0; ct < CT_ROW; ct++){
                for (int c = 0; c < K_TILE / 8; c++){
                    for (int m = 0; m < M_PER_CT; m++){
                        A_DATATYPE *p = AVec + (mt * M_PER_CT * CT_ROW + ct * M_PER_CT + m) * K + kt * K_TILE + c * 8;
                        memcpy(bufA, p, 8 * sizeof(A_DATATYPE));
                        bufA += 8;
                    } 
                }
            }
        }
    }
    // for (int ct = 0; ct < CT_ROW; ct++){
    //         // for each tile of A
    //         for (int c = 0; c < K_TILE / 8; c++){
    //             for (int m = 0; m < M_PER_ROUND; m++){
    //                 A_DATATYPE *p = AVec + mt * M_PER_ROUND * K + c * 8 + m * K;
    //                 memcpy(bufA, p, 16);
    //                 bufA += 8;
    //             }
    //         }
    //     }
    // }
}

void load_X(B_DATATYPE* bufB, B_DATATYPE* B_f){
// Concat three X vectors;
// Very slow opertions, need 
    int k_tiles = K / K_TILE;
    int chunk_size = 8 * K_CHUNK;
    int strides = N_TILE / 8;
    int chunks = K_TILE / K_CHUNK;

    transpose(B_f, bufB, K, N);
    return;
// 
    int i = 0;
    for (int nt = 0; nt < N / N_TILE; nt++){
        for (int kt = 0; kt < k_tiles; kt++){
            int row_offset = kt * K_TILE;
            for (int s = 0; s < strides; s++){
                for (int c = 0; c < chunks; c++){
                    // Part 1: Bf
                    for (int row = 0; row < K_CHUNK; row++){
                        int rr = c * K_CHUNK + row + row_offset;
                        int cc = (nt * strides + s) * 8;
                        B_DATATYPE* p = B_f + rr * N + cc;
                        memcpy(bufB, p, 8 * sizeof(B_DATATYPE));
                        bufB += 8;
                    }
                }
            }
        }
    }
}

void read_Y(C_DATATYPE* bufC, C_DATATYPE* CVec){
    memcpy(CVec, bufC, M * N * sizeof(C_DATATYPE));
    return;
    int i = 0;
    for (int nt = 0; nt < N / N_TILE; nt++){
        for (int mt = 0; mt < M / M_PER_CT; mt++){
            for (int m = 0; m < M_PER_CT; m++){
                memcpy(CVec + (mt * M_PER_CT + m) * N + nt * N_TILE, bufC, N_TILE * sizeof(A_DATATYPE));
                bufC += N_TILE;
                // for (int n = 0; n < N; n++){
                //     CVec[(mt * M_PER_ROUND + m) * N + n] = bufC[i++];
                // }
            }
        }
    }
}

#endif

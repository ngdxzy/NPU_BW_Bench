#ifndef __FUNCTIONS_HPP__
#define __FUNCTIONS_HPP__
#include "typedef.hpp"
#include <immintrin.h>
#include <vector>
#include <iostream>
#include <cmath>

#define USEAVX2 1

#if USEAVX2
inline __m256 bf16_to_fp32(__m128i bf16_vals);

// Convert float32 to bfloat16
inline __m128i fp32_to_bf16(__m256 fp32_vals);
#endif

void rms_norm(vdtype& y, const vdtype& x, const vdtype& w, dtype epsilon = (dtype)0.00001);

dtype inline exponential(dtype x);
void exponential(vdtype& y, const vdtype& x);

dtype inline sigmoid(dtype x);
void sigmoid(vdtype& y, const vdtype& x);

dtype inline logsigmoid(dtype x);
void logsigmoid(vdtype& y, const vdtype& x, dtype scale = (dtype)1.0);

dtype inline swish(dtype x);
void swish(vdtype& y, const vdtype& x);

void linear(vdtype& y, const vdtype& w, const vdtype& x);

void linear(vdtype& y, const vdtype& w, const vdtype& x, const vdtype& b);

void gated_linear_attention(vdtype& y, vdtype& s, const vdtype& q, const vdtype& k, const vdtype& v, const vdtype& alpha, int n_head);

void swiGlu(vdtype& y, const vdtype& gate, const vdtype& x);

void residual_split(vdtype& residual, const vdtype& x);

void dot(vdtype& y, const vdtype& w, const vdtype& x, int feature_dim);

void odot_rowwise(vdtype& y, const vdtype& A, const vdtype& B);
void odot_colwise(vdtype& y, const vdtype& A, const vdtype& B);

void odot(vdtype& y, dtype A, vdtype& B);

void add(vdtype& y, const vdtype& A, const vdtype& B);
void add(vector<int32_t>& y, const vector<int32_t>& A, const vector<int32_t>& B);


void linear_quantized(vdtype& y, const vector<int8_t>& wb, const vdtype& swb, const vdtype& x);

void linear_quantized(vdtype& y, const vector<int8_t>& wb, const vdtype& swb, const vdtype& x, const vdtype& b);

#endif

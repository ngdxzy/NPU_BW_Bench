#include "functions.hpp"
#include <math.h>

#if USEAVX2

// Convert bfloat16 to float32
inline __m256 bf16_to_fp32(__m128i bf16_vals) {
    __m256i expanded = _mm256_cvtepu16_epi32(bf16_vals); // Extend to 32-bit
    return _mm256_castsi256_ps(_mm256_slli_epi32(expanded, 16));  // Shift to float position
}

// Convert float32 to bfloat16
inline __m128i fp32_to_bf16(__m256 fp32_vals) {
    __m256i rounded = _mm256_srli_epi32(_mm256_castps_si256(fp32_vals), 16);  // Truncate lower bits
    return _mm_packus_epi32(_mm256_extracti128_si256(rounded, 0),
                            _mm256_extracti128_si256(rounded, 1));
}

// Element-wise exponential function for a single bfloat16 value
// inline dtype exponential(dtype x) {
//     float temp = std::exp((float)x);
//     return (dtype)((*((uint32_t*)&temp)) >> 16);  // Convert to BF16 by truncation
// }
// Approximate exp function using Taylor series expansion
inline __m256 exp_avx256(__m256 x) {
    // Constants
   const float LOG2E = 1.44269504088896340736f;  // 1/ln(2)
    const float EXP_HI = 88.3762626647949f;       // Upper limit to avoid overflow
    const float EXP_LO = -88.3762626647949f;      // Lower limit to avoid underflow
    const float LN2_HI = 0.6931471805f;           // log(2) high part
    const float LN2_LO = 1.9082149292705877e-10f; // log(2) low part (correction)

    // Polynomial coefficients for Taylor expansion
    const float P0 = 1.9875691500e-4f;
    const float P1 = 1.3981999507e-3f;
    const float P2 = 8.3334519073e-3f;
    const float P3 = 4.1665795894e-2f;
    const float P4 = 1.6666665459e-1f;
    const float P5 = 5.0000001201e-1f;

    // Load constants into AVX registers
    __m256 log2e = _mm256_set1_ps(LOG2E);
    __m256 ln2_hi = _mm256_set1_ps(LN2_HI);
    __m256 ln2_lo = _mm256_set1_ps(LN2_LO);
    __m256 exp_hi = _mm256_set1_ps(EXP_HI);
    __m256 exp_lo = _mm256_set1_ps(EXP_LO);

    // Clamp x to avoid overflow/underflow
    x = _mm256_max_ps(_mm256_min_ps(x, exp_hi), exp_lo);

    // Compute exponent and remainder
    __m256 fx = _mm256_mul_ps(x, log2e);
    __m256i n = _mm256_cvtps_epi32(fx);  // n = floor(fx)
    fx = _mm256_cvtepi32_ps(n);

    // Reduce x: r = x - n * log(2)
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(fx, ln2_hi));
    r = _mm256_sub_ps(r, _mm256_mul_ps(fx, ln2_lo));

    // Evaluate polynomial approximation of exp(r)
    __m256 poly = _mm256_set1_ps(P0);
    poly = _mm256_fmadd_ps(r, poly, _mm256_set1_ps(P1));
    poly = _mm256_fmadd_ps(r, poly, _mm256_set1_ps(P2));
    poly = _mm256_fmadd_ps(r, poly, _mm256_set1_ps(P3));
    poly = _mm256_fmadd_ps(r, poly, _mm256_set1_ps(P4));
    poly = _mm256_fmadd_ps(r, poly, _mm256_set1_ps(P5));
    poly = _mm256_fmadd_ps(r, poly, _mm256_set1_ps(1.0f));

    // Compute final result: 2^n * exp(r)
    __m256 result = _mm256_scalef_ps(poly, _mm256_cvtepi32_ps(n));

    return result;
}

// Vectorized exponential computation using AVX2
void exponential(vdtype& y, const vdtype& x) {
    int n = x.size();
    int simd_width = 8;  // AVX can process 8 bfloat16 elements at a time

    for (int i = 0; i < n; i += simd_width) {
        // Load 8 BF16 values from input
        __m128i bf16_vals = _mm_loadu_si128((__m128i*)&x[i]);

        // Convert BF16 to FP32
        __m256 fp32_vals = bf16_to_fp32(bf16_vals);

        // Compute exponential
        __m256 result_fp32 = exp_avx256(fp32_vals);

        // Convert FP32 back to BF16
        __m128i result_bf16 = fp32_to_bf16(result_fp32);

        // Store result
        _mm_storeu_si128((__m128i*)&y[i], result_bf16);
    }
}


void rms_norm(vdtype& y, const vdtype& x, const vdtype& w, dtype epsilon) {
    assert(x.size() == y.size());
    int n = x.size();
    const int simd_width = 8;

    __m256 sum_xx_vec = _mm256_setzero_ps();
    
    // Step 1: Compute sum of squares using AVX
    int i = 0;
    for (; i <= n - simd_width; i += simd_width) {
        __m128i bf16_vals = _mm_loadu_si128((__m128i*)&x[i]);

        // Convert BF16 to FP32
        __m256 fp32_vals = bf16_to_fp32(bf16_vals);
        sum_xx_vec = _mm256_fmadd_ps(fp32_vals, fp32_vals, sum_xx_vec);  // sum_xx += x[i] * x[i];
    }

    // Horizontal sum of the AVX register
    float sum_xx_arr[simd_width];
    _mm256_storeu_ps(sum_xx_arr, sum_xx_vec);
    accdtype sum_xx = sum_xx_arr[0] + sum_xx_arr[1] + sum_xx_arr[2] + sum_xx_arr[3] +
                      sum_xx_arr[4] + sum_xx_arr[5] + sum_xx_arr[6] + sum_xx_arr[7];

    // Process remaining elements (tail case)
    for (; i < n; ++i) {
        sum_xx += x[i] * x[i];
    }

    sum_xx /= (accdtype)n;
    accdtype rms_x = sqrtf(sum_xx + epsilon);

    // Step 2: Compute element-wise operation y[i] = w[i] * x[i] / rms_x
    __m256 rms_x_vec = _mm256_set1_ps(rms_x);
    for (i = 0; i <= n - simd_width; i += simd_width) {
        __m128i bf16_vals = _mm_loadu_si128((__m128i*)&x[i]);
        __m256 fp32_vals = bf16_to_fp32(bf16_vals);
        __m128i bf16_vals_w = _mm_loadu_si128((__m128i*)&w[i]);
        __m256 fp32_vals_w = bf16_to_fp32(bf16_vals_w);
        __m256 y_vec = _mm256_div_ps(_mm256_mul_ps(fp32_vals_w, fp32_vals), rms_x_vec);
        __m128i result_bf16 = fp32_to_bf16(y_vec);
        _mm_storeu_si128((__m128i*)&y[i], result_bf16);
    }

    // Process remaining elements (tail case)
    for (; i < n; ++i) {
        y[i] = (dtype)(w[i] * x[i] / rms_x);
    }
}

inline dtype scalar_logsigmoid(dtype x) {
    return (dtype)(-log1p(expf(-(float)x)));  // log1p(exp(-x)) = log(1 + exp(-x))
}

inline __m256 log_avx(__m256 x) {
    // Constants for logarithm approximation
    const __m256i exp_mask = _mm256_set1_epi32(0x7F800000);  // Mask to extract exponent bits
    const __m256i mant_mask = _mm256_set1_epi32(0x007FFFFF);  // Mask to extract mantissa bits
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 log2_e = _mm256_set1_ps(1.4426950408889634f);  // 1/log(2)
    const __m256 c1 = _mm256_set1_ps(0.6931471805f);  // log(2) high part
    const __m256 c2 = _mm256_set1_ps(1.9082149292705877e-10f); // log(2) low part

    // Polynomial coefficients for log approximation
    const __m256 P0 = _mm256_set1_ps(0.9999997f);
    const __m256 P1 = _mm256_set1_ps(-0.4999999f);
    const __m256 P2 = _mm256_set1_ps(0.3333333f);
    const __m256 P3 = _mm256_set1_ps(-0.2500000f);
    const __m256 P4 = _mm256_set1_ps(0.1999999f);
    const __m256 P5 = _mm256_set1_ps(-0.1666666f);

    // Handle special cases (x <= 0) by replacing with a small positive value to avoid domain errors
    __m256 min_val = _mm256_set1_ps(1e-30f);
    x = _mm256_max_ps(x, min_val);

    // Extract exponent and mantissa
    __m256i xi = _mm256_castps_si256(x);
    __m256i exponent = _mm256_srli_epi32(_mm256_and_si256(xi, exp_mask), 23); // Extract exponent
    __m256i mantissa = _mm256_and_si256(xi, mant_mask); // Extract mantissa

    // Normalize mantissa to the range [0.5, 1)
    mantissa = _mm256_or_si256(mantissa, _mm256_castps_si256(one));
    __m256 m = _mm256_castsi256_ps(mantissa);

    // Convert exponent to float and adjust
    __m256 e = _mm256_cvtepi32_ps(_mm256_sub_epi32(exponent, _mm256_set1_epi32(127)));

    // Polynomial approximation of log(m)
    __m256 m_minus_one = _mm256_sub_ps(m, one);
    __m256 u = _mm256_div_ps(m_minus_one, _mm256_add_ps(m, one));
    __m256 u2 = _mm256_mul_ps(u, u);

    __m256 poly = _mm256_fmadd_ps(u2, P5, P4);
    poly = _mm256_fmadd_ps(u2, poly, P3);
    poly = _mm256_fmadd_ps(u2, poly, P2);
    poly = _mm256_fmadd_ps(u2, poly, P1);
    poly = _mm256_fmadd_ps(u2, poly, P0);
    poly = _mm256_mul_ps(poly, u);

    // Compute final logarithm value: log(x) = e * log(2) + log(m)
    __m256 log_val = _mm256_fmadd_ps(e, c1, poly);
    log_val = _mm256_add_ps(log_val, _mm256_mul_ps(e, c2));

    return log_val;
}

// AVX approximation of log1p(x) = log(1 + x)
inline __m256 log1p_avx(__m256 x) {
    // Constants for polynomial approximation and handling
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 log2_e = _mm256_set1_ps(1.4426950408889634f);  // 1/log(2)
    const __m256 ln2 = _mm256_set1_ps(0.6931471805f);            // log(2)

    // Polynomial coefficients for log(1 + x) approximation (up to x^6 term)
    const __m256 C1 = _mm256_set1_ps(-0.3333333f);
    const __m256 C2 = _mm256_set1_ps(0.2000000f);
    const __m256 C3 = _mm256_set1_ps(-0.1428571f);
    const __m256 C4 = _mm256_set1_ps(0.1111111f);
    const __m256 C5 = _mm256_set1_ps(-0.0909091f);
    const __m256 C6 = _mm256_set1_ps(0.0769230f);

    // Handle small values with the polynomial approximation
    __m256 small_mask = _mm256_cmp_ps(x, half, _CMP_LE_OQ); // x <= 0.5
    __m256 x_small = _mm256_div_ps(x, _mm256_add_ps(one, x));
    __m256 x_small_sq = _mm256_mul_ps(x_small, x_small);

    __m256 poly = _mm256_fmadd_ps(x_small_sq, C6, C5);
    poly = _mm256_fmadd_ps(x_small_sq, poly, C4);
    poly = _mm256_fmadd_ps(x_small_sq, poly, C3);
    poly = _mm256_fmadd_ps(x_small_sq, poly, C2);
    poly = _mm256_fmadd_ps(x_small_sq, poly, C1);
    poly = _mm256_mul_ps(poly, x_small_sq);
    poly = _mm256_fmadd_ps(x_small, poly, x_small);

    __m256 result_small = _mm256_mul_ps(poly, _mm256_set1_ps(2.0f));

    // Handle larger values with decomposition: log1p(x) = log(x + 1) = e * log(2) + log(mantissa)
    __m256 large_x = _mm256_add_ps(x, one);
    __m256 exp_f = _mm256_mul_ps(log2_e, log_avx(large_x));

    // Select between polynomial and standard computation
    return _mm256_blendv_ps(exp_f, result_small, small_mask);
}


void logsigmoid(vdtype& y, const vdtype& x, dtype scale) {
    int n = x.size();
    assert(y.size() == x.size());

    __m256 scale_vec = _mm256_set1_ps(scale);
    __m256 one_vec = _mm256_set1_ps(1.0f);

    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m128i bf16_vals = _mm_loadu_si128((__m128i*)&x[i]);
        __m256 fp32_vals = bf16_to_fp32(bf16_vals);

        // Compute exp(-x) using AVX
        __m256 neg_x_vec = _mm256_sub_ps(_mm256_setzero_ps(), fp32_vals);
        __m256 exp_neg_x_vec = exp_avx256(neg_x_vec);

        // logsigmoid(x) = -log1p(exp(-x))
        __m256 log1p_exp = log1p_avx(exp_neg_x_vec);
        __m256 logsigmoid_vec = _mm256_sub_ps(_mm256_setzero_ps(), log1p_exp);

        // Apply scaling: y[i] = scale * logsigmoid(x[i])
        __m256 result_vec = _mm256_mul_ps(scale_vec, logsigmoid_vec);

        // Store results
        __m128i result_bf16 = fp32_to_bf16(result_vec);
        _mm_storeu_si128((__m128i*)&y[i], result_bf16);
    }

    // Handle remaining elements (tail case)
    for (; i < n; ++i) {
        y[i] = scale * scalar_logsigmoid(x[i]);
    }
}

dtype inline sigmoid(dtype x){
    return (dtype)(1.0 / (1.0 + expf(-(float)x)));
}

inline __m256 sigmoid_avx(__m256 x){
    __m256 one_vec = _mm256_set1_ps(1.0f);
    __m256 neg_x_vec = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg_x_vec = exp_avx256(neg_x_vec);
    return _mm256_div_ps(one_vec, _mm256_add_ps(one_vec, exp_neg_x_vec));
}

void sigmoid(vdtype& y, const vdtype& x){
    int n = x.size();
    int simd_width = 8;
    __m256 one_vec = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i <= n - simd_width; i += simd_width){
        __m128i bf16_vals = _mm_loadu_si128((__m128i*)&x[i]);
        __m256 fp32_vals = bf16_to_fp32(bf16_vals);
        __m256 result_vec = sigmoid_avx(fp32_vals);
        __m128i result_bf16 = fp32_to_bf16(result_vec);
        _mm_storeu_si128((__m128i*)&y[i], result_bf16);
    }

    // Handle remaining elements (tail case)
    for (; i < n; ++i) {
        y[i] = sigmoid(x[i]);
    }
}
void linear(vdtype& y, const vdtype& w, const vdtype& x) {
    int in_features = x.size();
    int out_features = y.size();
    const int simd_size = 8;

    assert((in_features * out_features) == w.size());

    int v = 0;
    for (int row = 0; row < out_features; row++) {
        __m256 sum_vec = _mm256_setzero_ps();  // Initialize sum vector to zero

        int col = 0;
        for (; col <= in_features - simd_size; col += simd_size) {
            // Load 8 values from weights and input vectors
            __m128i bf16_vals = _mm_loadu_si128((__m128i*)&w[v]);
            __m256 fp32_vals = bf16_to_fp32(bf16_vals);
            __m128i bf16_vals_x = _mm_loadu_si128((__m128i*)&x[col]);
            __m256 fp32_vals_x = bf16_to_fp32(bf16_vals_x); 

            // Multiply and accumulate
            sum_vec = _mm256_fmadd_ps(fp32_vals, fp32_vals_x, sum_vec);  // sum += w * x

            v += simd_size;
        }

        // Reduce the sum vector to a scalar sum
        float sum_arr[simd_size];
        _mm256_storeu_ps(sum_arr, sum_vec);
        accdtype sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
                       sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];

        // Handle any remaining elements (tail processing)
        for (; col < in_features; col++) {
            sum += accdtype(w[v++] * x[col]);
        }

        y[row] = (dtype)sum;
    }
}

void linear(vdtype& y, const vdtype& w, const vdtype& x, const vdtype& b){
    int in_features = x.size();
    int out_features = y.size();
    
    assert(y.size() == b.size());
    assert((in_features * out_features) == w.size());

    int v = 0;
    for (int row = 0; row < out_features; row++) {
        __m256 sum_vec = _mm256_setzero_ps();  // Initialize sum vector to zero

        int col = 0;
        for (; col <= in_features - 8; col += 8) {
            // Load 8 values from weights and input vectors
            __m128i bf16_vals = _mm_loadu_si128((__m128i*)&w[v]);
            __m256 fp32_vals = bf16_to_fp32(bf16_vals);
            __m128i bf16_vals_x = _mm_loadu_si128((__m128i*)&x[col]);
            __m256 fp32_vals_x = bf16_to_fp32(bf16_vals_x); 

            // Multiply and accumulate
            sum_vec = _mm256_fmadd_ps(fp32_vals, fp32_vals_x, sum_vec);  // sum += w * x

            v += 8;
        }

        // Reduce the sum vector to a scalar sum
        float sum_arr[8];
        _mm256_storeu_ps(sum_arr, sum_vec);
        accdtype sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
                       sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];

        // Handle any remaining elements (tail processing)
        for (; col < in_features; col++) {
            sum += accdtype(w[v++] * x[col]);
        }

        y[row] = (dtype)(sum + b[row]);
    }

}



dtype inline swish(dtype x){
    return (dtype)(x * sigmoid(x));
}

inline __m256 swish_avx(__m256 x){
    return _mm256_mul_ps(x, sigmoid_avx(x));
}

void swish(vdtype& y, const vdtype& x){
    const int simd_width = 8;
    int n = x.size();
    int i = 0;
    
    for (; i <= n - simd_width; i += simd_width){
        __m128i bf16_vals = _mm_loadu_si128((__m128i*)&x[i]);
        __m256 fp32_vals = bf16_to_fp32(bf16_vals);
        __m256 result_vec = swish_avx(fp32_vals);
        __m128i result_bf16 = fp32_to_bf16(result_vec);
        _mm_storeu_si128((__m128i*)&y[i], result_bf16);
    }
}

void swiGlu(vdtype& y, const vdtype& gate, const vdtype& x){
    int n = gate.size();
    assert(y.size() == n);
    assert(x.size() == n);
    const int simd_width = 8;
    
    int i = 0;
    for (; i <= n - simd_width; i += simd_width){
        __m128i bf16_vals_x = _mm_loadu_si128((__m128i*)&x[i]);
        __m256 fp32_vals_x = bf16_to_fp32(bf16_vals_x);
        __m128i bf16_vals_gate = _mm_loadu_si128((__m128i*)&gate[i]);
        __m256 fp32_vals_gate = bf16_to_fp32(bf16_vals_gate);
        __m256 result_vec = _mm256_mul_ps(fp32_vals_x, swish_avx(fp32_vals_gate));
        __m128i result_bf16 = fp32_to_bf16(result_vec);
        _mm_storeu_si128((__m128i*)&y[i], result_bf16);
    }
    for (; i < n; i++){
        y[i] = (dtype)(x[i] * swish(gate[i]));
    }
}


void add(vdtype& y, const vdtype& A, const vdtype& B){
    int n = A.size();
    assert(y.size() == n);
    assert(A.size() == B.size());

    const int simd_width = 8;
    int i = 0;
    for (; i <= n - simd_width; i += simd_width){
        __m128i bf16_vals_A = _mm_loadu_si128((__m128i*)&A[i]);
        __m256 fp32_vals_A = bf16_to_fp32(bf16_vals_A);
        __m128i bf16_vals_B = _mm_loadu_si128((__m128i*)&B[i]);
        __m256 fp32_vals_B = bf16_to_fp32(bf16_vals_B);
        __m256 result_vec = _mm256_add_ps(fp32_vals_A, fp32_vals_B);
        __m128i result_bf16 = fp32_to_bf16(result_vec);
        _mm_storeu_si128((__m128i*)&y[i], result_bf16);
    }
    for (; i < n; i++){
        y[i] = (dtype)(A[i] + B[i]);
    }
}

void add(vector<int32_t>& y, const vector<int32_t>& A, const vector<int32_t>& B){
    int n = A.size();
    assert(y.size() == n);
    assert(A.size() == B.size());

    for (int i = 0; i < n; i++){
        y[i] = A[i] + B[i];
    }
}


#else
dtype inline exponential(dtype x){
    return (dtype)(expf((float)x));
}

void exponential(vdtype& y, const vdtype& x){
    int n = x.size();
    for (int i = 0; i < n; i++){
        y[i] = exponential(x[i]);
    }
}


void rms_norm(vdtype& y, const vdtype& x, const vdtype& w, dtype epsilon){
    assert(x.size() == y.size());

    int n = x.size();
    accdtype sum_xx = 0.0f;
    for (int i = 0; i < n; i++){
        sum_xx += x[i] * x[i];
    }
    sum_xx /= (accdtype)n;
    accdtype rms_x = sqrtf(sum_xx + epsilon);

    for (int i = 0; i < n; i++){
        y[i] = (dtype)(w[i] * x[i] / rms_x);
    }
}



dtype inline logsigmoid(dtype x){
    return (dtype)(-logf(1.0 + expf(-(float)x)));
}

void logsigmoid(vdtype& y, const vdtype& x, dtype scale){
    int n = x.size();
    for (int i = 0; i < n; i++){
        y[i] = scale * logsigmoid(x[i]);
    }
}


dtype inline sigmoid(dtype x){
    return (dtype)(1.0 / (1.0 + expf(-(float)x)));
}

void sigmoid(vdtype& y, const vdtype& x){
    int n = x.size();
    for (int i = 0; i < n; i++){
        y[i] = sigmoid(x[i]);
    }
}

void linear(vdtype& y, const vdtype& w, const vdtype& x){
    int in_features = x.size();
    int out_features = y.size();
    // std::cout << "in_features: " << in_features << std::endl;
    // std::cout << "out_features: " << out_features << std::endl;
    // std::cout << "w.size(): " << w.size() << std::endl;
    assert((in_features * out_features) == w.size());
    int v = 0;
    for (int row = 0; row < out_features; row++){
        accdtype sum = 0;
        for (int col = 0; col < in_features; col++){
            sum = sum + accdtype(w[v++] * x[col]);
        }
        y[row] = (dtype)sum;
    }
}

void linear(vdtype& y, const vdtype& w, const vdtype& x, const vdtype& b){
    int in_features = x.size();
    int out_features = y.size();
    assert(y.size() == b.size());
    assert((in_features * out_features) == w.size());
    int v = 0;
    for (int row = 0; row < out_features; row++){
        accdtype sum = (accdtype)b[row];
        for (int col = 0; col < in_features; col++){
            sum = sum + accdtype(w[v++] * x[col]);
        }
        y[row] = (dtype)sum;
    }
}


dtype inline swish(dtype x){
    return (dtype)(x * sigmoid(x));
}

void swish(vdtype& y, const vdtype& x){
    int n = x.size();
    for (int i = 0; i < n; i++){
        y[i] = swish(x[i]);
    }
}

void swiGlu(vdtype& y, const vdtype& gate, const vdtype& x){
    int n = gate.size();
    assert(y.size() == n);
    assert(x.size() == n);

    for (int i = 0; i < n; i++){
        y[i] = (dtype)(x[i] * swish(gate[i]));
    }
}


void add(vdtype& y, const vdtype& A, const vdtype& B){
    int n = A.size();
    assert(y.size() == n);
    assert(A.size() == B.size());

    for (int i = 0; i < n; i++){
        y[i] = (dtype)(A[i] + B[i]);
    }
}

#endif




void gated_linear_attention(vdtype& y, vdtype& s, const vdtype& q, const vdtype& k, const vdtype& v, const vdtype& alpha, int n_head = 1){
    int Dv = y.size();
    int Dk = q.size();
    int dv = Dv / n_head;
    int dk = Dk / n_head;

    assert(s.size() == (Dk * dv));
    for (int h = 0; h < n_head; h++){
        int yv_offset = dv * h;
        int s_offset = dk * dv * h;
        int qka_offset = dk * h;

        for (int ik = 0; ik < dk; ik++){
            for (int iv = 0; iv < dv; iv++){
                s[s_offset + ik * dv + iv] = alpha[qka_offset + ik] * s[s_offset + ik * dv + iv] + k[qka_offset + ik] * v[yv_offset + iv];
            }
        }

        for (int iv = 0; iv < dv; iv++){
            accdtype sum = 0;
            for (int ik = 0; ik < dk; ik++){
                sum += q[qka_offset + ik] * s[s_offset + ik * dv + iv];
            }
            y[yv_offset + iv] = (dtype)sum;
        }
    }
}


void residual_split(vdtype& residual, const vdtype& x){
    size_t size = x.size() * sizeof(dtype);

    if (residual.data() == nullptr){
        residual.acquire(x.size());
    }
    else{
        assert(residual.size() == x.size());
    }

    memcpy(residual.data(), x.data(), size);
}

void dot(vdtype& y, const vdtype& A, const vdtype& B, int feature_dim){
    
    int m = A.size() / feature_dim;
    int n = B.size() / feature_dim;

    assert(m > 0);
    assert(n > 0);

    // A is m x feature_dim
    // B is feature_dim x n
    // y is m x n
    assert(y.size() == m * n);

    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            accdtype sum = 0;
            for (int k = 0; k < feature_dim; k++){
                sum += A[i * feature_dim + k] * B[k * n + j];
            }
            y[i * n + j] = (dtype)sum;
        }
    }
}

void odot_rowwise(vdtype& y, const vdtype& A, const vdtype& B){
    int m = A.size();
    int n = B.size() / m;

    assert(y.size() == m * n);


    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            y[i * n + j] = (dtype)(A[i] * B[i * n + j]);
        }
    }
}

void odot_colwise(vdtype& y, const vdtype& A, const vdtype& B){
    int m = A.size();
    int n = B.size() / m;

    for (int j = 0; j < n; j++){
        for (int i = 0; i < m; i++){
            y[i * n + j] = (dtype)(A[i] * B[j * m + i]);
        }
    }
}

void odot(vdtype& y, dtype A, const vdtype& B){
    int n = B.size();

    for (int i = 0; i < n; i++){
        y[i] = (dtype)(A * B[i]);
    }
}

void linear_quantized(vdtype& y, const vector<int8_t>& wb, const vdtype& swb, const vdtype& x){
    int in_features = x.size();
    int out_features = y.size();
    
    const dtype scale = 1.0 / 127;
    // find max_x
    dtype max_x = 0;
    for (int i = 0; i < in_features; i++){
        if (abs(x[i]) > max_x){
            max_x = abs(x[i]);
        }
    }
    dtype sxb = max_x * scale;
    vector<int8_t> xb(in_features);
    for (int i = 0; i < in_features; i++){
        xb[i] = (int8_t)(x[i] / sxb);
    }
    for (int row = 0; row < out_features; row++) {
        __m256i acc = _mm256_setzero_si256();  // Initialize accumulator to zero

        int col = 0;
        for (; col < in_features; col += 16) {

            // Unpack and multiply (8-bit -> 16-bit multiplication)
            __m256i a_signed = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&wb[row * in_features + col])));
            __m256i b_signed = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&xb[col])));

            // Multiply 16-bit integers and add to accumulator
            __m256i mul_result = _mm256_madd_epi16(a_signed, b_signed);

            // Accumulate into 32-bit integers
            acc = _mm256_add_epi32(acc, mul_result);
        }

        // Reduce across 256-bit registers (horizontal addition)
        __m128i acc_high = _mm256_extracti128_si256(acc, 1);
        __m128i acc_low = _mm256_castsi256_si128(acc);
        __m128i result = _mm_add_epi32(acc_low, acc_high);

        // Store final sum for this row
        int32_t sum = _mm_extract_epi32(result, 0) + _mm_extract_epi32(result, 1) +
               _mm_extract_epi32(result, 2) + _mm_extract_epi32(result, 3);
        y[row] = (dtype)(sum * sxb * swb[row]);
    }
    // for (int i = 0; i < out_features; i++){
    //     int32_t sum = 0;
    //     for (int j = 0; j < in_features; j++){
    //         sum += wb[i * in_features + j] * xb[j];
    //     }
    //     y[i] = (dtype)(sum * sxb * swb[i]);
    // }
}

void linear_quantized(vdtype& y, const vector<int8_t>& wb, const vdtype& swb, const vdtype& x, const vdtype& b){
        int in_features = x.size();
    int out_features = y.size();
    
    const dtype scale = 1.0 / 127;
    // find max_x
    dtype max_x = 0;
    for (int i = 0; i < in_features; i++){
        if (abs(x[i]) > max_x){
            max_x = abs(x[i]);
        }
    }
    dtype sxb = max_x * scale;
    vector<int8_t> xb(in_features);
    for (int i = 0; i < in_features; i++){
        xb[i] = (int8_t)(x[i] / sxb);
    }

    for (int row = 0; row < out_features; row++) {
        __m256i acc = _mm256_setzero_si256();  // Initialize accumulator to zero

        int col = 0;
        for (; col < in_features; col += 16) {

            // Unpack and multiply (8-bit -> 16-bit multiplication)
            __m256i a_signed = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&wb[row * in_features + col])));
            __m256i b_signed = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&xb[col])));

            // Multiply 16-bit integers and add to accumulator
            __m256i mul_result = _mm256_madd_epi16(a_signed, b_signed);

            // Accumulate into 32-bit integers
            acc = _mm256_add_epi32(acc, mul_result);
        }

        // Reduce across 256-bit registers (horizontal addition)
        __m128i acc_high = _mm256_extracti128_si256(acc, 1);
        __m128i acc_low = _mm256_castsi256_si128(acc);
        __m128i result = _mm_add_epi32(acc_low, acc_high);

        // Store final sum for this row
        int32_t sum = _mm_extract_epi32(result, 0) + _mm_extract_epi32(result, 1) +
               _mm_extract_epi32(result, 2) + _mm_extract_epi32(result, 3);
        y[row] = (dtype)(sum * sxb * swb[row]) + b[row];
    }
}

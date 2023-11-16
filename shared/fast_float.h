/**
 * Copyright (C) 2023 by Michal Vlnas <ivlnas@fit.vutbr.cz>
 *
 * Permission to use, copy, modify, and/or distribute this
 * software for any purpose with or without fee is hereby granted.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
 * WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
 * THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR
 * CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
 * CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
 * OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef IMPORTANCE_DISTRIBUTIONS_FAST_FLOAT_H
#define IMPORTANCE_DISTRIBUTIONS_FAST_FLOAT_H

#include <shared/config.h>
#include <xmmintrin.h>
#include <immintrin.h>

namespace Importance
{
namespace Ff
{

namespace FastMath
{

#define ALIGN16_BEG __attribute__ ((aligned (16)))
#define ALIGN16_END

/* declare some SSE constants -- why can't I figure a better way to do that? */
#define _PS_CONST(Name, Val)                                            \
     static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PI32_CONST(Name, Val)                                            \
    static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PS_CONST_TYPE(Name, Type, Val)                                 \
    static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }

_PS_CONST(1, 1.0f);
_PS_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
//_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS_CONST_TYPE(sign_mask, int, static_cast<int>(0x80000000));
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS_CONST(cephes_log_q1, -2.12194440e-4f);
_PS_CONST(cephes_log_q2, 0.693359375f);

_PS_CONST(exp_hi, 88.3762626647949f);
_PS_CONST(exp_lo, -88.3762626647949f);

_PS_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS_CONST(cephes_exp_C1, 0.693359375f);
_PS_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1f);

_PS_CONST(minus_cephes_DP1, -0.78515625f);
_PS_CONST(minus_cephes_DP2, -2.4187564849853515625e-4f);
_PS_CONST(minus_cephes_DP3, -3.77489497744594108e-8f);
_PS_CONST(sincof_p0, -1.9515295891E-4f);
_PS_CONST(sincof_p1,  8.3321608736E-3f);
_PS_CONST(sincof_p2, -1.6666654611E-1f);
_PS_CONST(coscof_p0,  2.443315711809948E-005f);
_PS_CONST(coscof_p1, -1.388731625493765E-003f);
_PS_CONST(coscof_p2,  4.166664568298827E-002f);
_PS_CONST(cephes_FOPI, 1.27323954473516f); // 4 / M_PI

void sincos_ps(__m128 x, __m128 *s, __m128 *c)
{
    __m128 xmm1, xmm2, xmm3 = _mm_setzero_ps(), sign_bit_sin, y;
    __m128i emm0, emm2, emm4;

    sign_bit_sin = x;
    /* take the absolute value */
    x = _mm_and_ps(x, *(__m128 *) _ps_inv_sign_mask);
    /* extract the sign bit (upper one) */
    sign_bit_sin = _mm_and_ps(sign_bit_sin, *(__m128 *) _ps_sign_mask);

    /* scale by 4/Pi */
    y = _mm_mul_ps(x, *(__m128 *) _ps_cephes_FOPI);

    /* store the integer part of y in emm2 */
    emm2 = _mm_cvttps_epi32(y);

    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = _mm_add_epi32(emm2, *(__m128i *) _pi32_1);
    emm2 = _mm_and_si128(emm2, *(__m128i *) _pi32_inv1);
    y = _mm_cvtepi32_ps(emm2);

    emm4 = emm2;

    /* get the swap sign flag for the sine */
    emm0 = _mm_and_si128(emm2, *(__m128i *) _pi32_4);
    emm0 = _mm_slli_epi32(emm0, 29);
    __m128 swap_sign_bit_sin = _mm_castsi128_ps(emm0);

    /* get the polynom selection mask for the sine*/
    emm2 = _mm_and_si128(emm2, *(__m128i *) _pi32_2);
    emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
    __m128 poly_mask = _mm_castsi128_ps(emm2);


    /* The magic pass: "Extended precision modular arithmetic" 
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    xmm1 = *(__m128 *) _ps_minus_cephes_DP1;
    xmm2 = *(__m128 *) _ps_minus_cephes_DP2;
    xmm3 = *(__m128 *) _ps_minus_cephes_DP3;
    xmm1 = _mm_mul_ps(y, xmm1);
    xmm2 = _mm_mul_ps(y, xmm2);
    xmm3 = _mm_mul_ps(y, xmm3);
    x = _mm_add_ps(x, xmm1);
    x = _mm_add_ps(x, xmm2);
    x = _mm_add_ps(x, xmm3);

    emm4 = _mm_sub_epi32(emm4, *(__m128i *) _pi32_2);
    emm4 = _mm_andnot_si128(emm4, *(__m128i *) _pi32_4);
    emm4 = _mm_slli_epi32(emm4, 29);
    __m128 sign_bit_cos = _mm_castsi128_ps(emm4);

    sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    /* Evaluate the first polynom  (0 <= x <= Pi/4) */
    __m128 z = _mm_mul_ps(x, x);
    y = *(__m128 *) _ps_coscof_p0;

    y = _mm_mul_ps(y, z);
    y = _mm_add_ps(y, *(__m128 *) _ps_coscof_p1);
    y = _mm_mul_ps(y, z);
    y = _mm_add_ps(y, *(__m128 *) _ps_coscof_p2);
    y = _mm_mul_ps(y, z);
    y = _mm_mul_ps(y, z);
    __m128 tmp = _mm_mul_ps(z, *(__m128 *) _ps_0p5);
    y = _mm_sub_ps(y, tmp);
    y = _mm_add_ps(y, *(__m128 *) _ps_1);

    /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

    __m128 y2 = *(__m128 *) _ps_sincof_p0;
    y2 = _mm_mul_ps(y2, z);
    y2 = _mm_add_ps(y2, *(__m128 *) _ps_sincof_p1);
    y2 = _mm_mul_ps(y2, z);
    y2 = _mm_add_ps(y2, *(__m128 *) _ps_sincof_p2);
    y2 = _mm_mul_ps(y2, z);
    y2 = _mm_mul_ps(y2, x);
    y2 = _mm_add_ps(y2, x);

    /* select the correct result from the two polynoms */
    xmm3 = poly_mask;
    __m128 ysin2 = _mm_and_ps(xmm3, y2);
    __m128 ysin1 = _mm_andnot_ps(xmm3, y);
    y2 = _mm_sub_ps(y2, ysin2);
    y = _mm_sub_ps(y, ysin1);

    xmm1 = _mm_add_ps(ysin1, ysin2);
    xmm2 = _mm_add_ps(y, y2);

    /* update the sign */
    *s = _mm_xor_ps(xmm1, sign_bit_sin);
    *c = _mm_xor_ps(xmm2, sign_bit_cos);
}

} // namespace FastMath

inline float log(const float x)
{
    return ::log(x);
}

inline void sincos(const Float in, Float &outSin, Float &outCos)
{
    __m128 temp1, temp2;
    FastMath::sincos_ps(_mm_set1_ps(in), &temp1, &temp2);
    outSin = *(float *) &temp1;
    outCos = *(float *) &temp2;
}

} // namespace Ff

inline __m256 scan_avx2(__m256 x)
{
    __m256 zero = _mm256_set1_ps(0);
    __m256 t0 = _mm256_permutevar8x32_ps(x, _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6));
    x = _mm256_add_ps(x, _mm256_blend_ps(t0, zero, 0x01));

    t0 = _mm256_permutevar8x32_ps(x, _mm256_setr_epi32(0, 0, 0, 1, 2, 3, 4, 5));
    x = _mm256_add_ps(x, _mm256_blend_ps(t0, zero, 0x03));

    t0 = _mm256_permutevar8x32_ps(x, _mm256_setr_epi32(0, 0, 0, 0, 0, 1, 2, 3));
    x = _mm256_add_ps(x, _mm256_blend_ps(t0, zero, 0x0F));

    return x;
}

inline void avx2_sum(float* arr, std::size_t n)
{
    std::size_t n8 = n & ~7u;
    __m256 offset = _mm256_setzero_ps();
    for (std::size_t i = 0; i < n8; i += 8)
    {
        __m256 x = _mm256_loadu_ps(&arr[i]);
        __m256 out = scan_avx2(x);
        out = _mm256_add_ps(out, offset);
        _mm256_storeu_ps(&arr[i], out);
        // broadcast last element
        offset = _mm256_permutevar8x32_ps(out, _mm256_set1_epi32(7));
    }

    // unaligned sequential finish
    if (!n8) n8 = 1;
    for (std::size_t i = n8; i < n; ++i)
        arr[i] += arr[i - 1];
}

} // namespace Importance

#endif //IMPORTANCE_DISTRIBUTIONS_FAST_FLOAT_H

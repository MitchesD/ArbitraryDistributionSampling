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

#ifndef IMPORTANCE_XORSHIFT_H__
#define IMPORTANCE_XORSHIFT_H__

#include <shared/config.h>
#include <glm/glm.hpp>
#include <NTL/GF2X.h>

#define INV_UNSIGNED_RAND_MAX 2.32831e-10f

namespace Importance
{

/**
 * @brief The XorShiftGenerator class
 *
 * This is a modified version of the XorShift generator from
 * <https://www.codeproject.com/Articles/5265915/XorShift-Jump-101-Part-2-Polynomial-Arithmetic>
 */
class XorShiftGenerator
{
private:
    const size_t STATE_SIZE_EXP  = 128;
    const size_t STATE_WORD_SIZE = CHAR_BIT * sizeof(uint32_t);
    const size_t STATE_SIZE      = STATE_SIZE_EXP / STATE_WORD_SIZE;

    using poly_t = NTL::GF2X;

public:
    using state_t = std::array<uint32_t, 4>;

    explicit XorShiftGenerator()
    {
        init_transition_polynomials();
        reset();
    }

    XorShiftGenerator(const XorShiftGenerator &rhs) : state(rhs.state), fwd_step_mod(rhs.fwd_step_mod),
        pre_tr_k(rhs.pre_tr_k)
    {
    }

    FORCE_INLINE Float next1D()
    {
        skip1D();
        return state[3] * INV_UNSIGNED_RAND_MAX;
    }

    FORCE_INLINE uint32_t next1DInt()
    {
        skip1D();
        return state[3];
    }

    FORCE_INLINE void skip1D()
    {
        uint32_t &x = state[0], &y = state[1], &z = state[2], &w = state[3];

        uint32_t t = x ^ (x << 11u);

        x = y; y = z; z = w;
        w = w ^ (w >> 19u) ^ (t ^ (t >> 8u));
    }

    static void skip1D(state_t& st)
    {
        uint32_t &x = st[0], &y = st[1], &z = st[2], &w = st[3];

        uint32_t t = x ^ (x << 11u);

        x = y; y = z; z = w;
        w = w ^ (w >> 19u) ^ (t ^ (t >> 8u));
    }

    glm::vec2 next2D()
    {
        return { next1D(), next1D() };
    }

    std::tuple<uint32_t, uint32_t> next2DInt()
    {
        return { next1DInt(), next1DInt() };
    }

    void skipN(uint64_t k)
    {
        if (k == 0)
            return;

        poly_t t_k;
        if (k < 255)
            t_k = pre_tr_k[k];
        else
            prepare_transition(t_k, k);
        do_transition(t_k);
    }

    void skip2D()
    {
        skip1D();
        skip1D();
    }

    void reset()
    {
        state[0] = 123456789;
        state[1] = 362436069;
        state[2] = 521288629;
        state[3] = 88675123;
    }

    void print_state()
    {
        std::cout << state[0] << " & " << state[1] << " & " << state[2] << " & " << state[3] << std::endl;
    }

    std::string get_state_string() const
    {
        return std::to_string(state[0]) + " & " + std::to_string(state[1])
                + " & " + std::to_string(state[2]) + " & " + std::to_string(state[3]);
    }

private:
    void init_state(state_t& st)
    {
        st[0] = 123456789;
        st[1] = 362436069;
        st[2] = 521288629;
        st[3] = 88675123;
    }

    void init_transition_polynomials()
    {
        pre_tr_k.clear();

        state_t st;
        init_state(st);

        const size_t N = 2 * STATE_SIZE_EXP;

        NTL::vec_GF2 vf(NTL::INIT_SIZE, N);

        for (size_t i = 0; i < N; i++)
        {
            skip1D(st);
            vf[i] = st[3] & 0x01ul;
        }

        NTL::GF2X fwd_step_poly;
        NTL::MinPolySeq(fwd_step_poly, vf, STATE_SIZE_EXP);
        NTL::build(fwd_step_mod, fwd_step_poly);

        // pre-compute transition up to 255
        pre_tr_k.reserve(255);
        for (uint64_t k = 0; k < 255; k++)
        {
            poly_t tr_k;
            NTL::GF2X x(1, 1);
            NTL::PowerMod(tr_k, x, k, fwd_step_mod);
            pre_tr_k.push_back(tr_k);
        }
    }

    void prepare_transition(poly_t& tr_k, uint64_t k)
    {
        NTL::GF2X x(1, 1);
        NTL::PowerMod(tr_k, x, k, fwd_step_mod);
    }

    static void add_state(state_t &x, const state_t &y)
    {
        for (size_t i = 0; i < x.size(); i++)
            x[i] = x[i] ^ y[i];
    }

    void horner(state_t &s, const poly_t &tr)
    {
        state_t tmp_state = s;

        tmp_state.fill(0);

        int i = NTL::deg(tr);

        if (i > 0)
        {
            for (; i > 0; i--)
            {
                if (NTL::coeff(tr, i) != 0)
                    add_state(tmp_state, s);

                skip1D(tmp_state);
            }

            if (NTL::coeff(tr, 0) != 0)
                add_state(tmp_state, s);
        }

        s = tmp_state;
    }

    void do_transition(const poly_t &tr)
    {
        horner(state, tr);
    }

    state_t state;

    NTL::GF2XModulus fwd_step_mod;
    std::vector<poly_t> pre_tr_k;
};

} // namespace Importance

#endif // IMPORTANCE_XORSHIFT_H__

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

#ifndef IMPORTANCE_MISC_DISTR_H__
#define IMPORTANCE_MISC_DISTR_H__

#include <shared/distribution.h>
#include <is/ibuffer.h>

namespace Importance
{

class NormalDistribution : public Distribution1D
{
public:
    NormalDistribution(Float mu_, Float si_, Float x_min_, Float x_max_)
        : Distribution1D(x_min_, x_max_), mu(mu_), si(si_) { }

    Float f(Float const& x) const final
    {
        Float exponent = -0.5f * ((x - mu) / si) * ((x - mu) / si);
        return 1.0f / (si * std::sqrt(2.0f * F_PI)) * std::exp(exponent);
    }

    Float max() const final
    {
        return f(mu);
    }

    Float cdf(Float const& x) const
    {
        float err = std::erf((x - mu) / (si * std::sqrt(2.0f)));
        return 0.5f * (1.0f + err);
    }

private:
    Float mu;
    Float si;
};

template<class TGenerator>
class INormalDistribution : public NormalDistribution
{
public:
    INormalDistribution(Float mu_, Float si_, Float x_min_, Float x_max_)
        : NormalDistribution(mu_, si_, x_min_, x_max_), seq_id(0) { }

    Float sample(AbstractImportanceBuffer<NormalDistribution, TGenerator, Float>* buffer, TGenerator* gen)
    {
        return buffer->sample1D(seq_id, gen, x_range, x_min);
    }

public:
    uint32_t seq_id;
};

class LandauDistribution : public Distribution1D
{
public:
    LandauDistribution(Float x_min_, Float x_max_) : Distribution1D(x_min_, x_max_) { }

    /**
     * Approximation of Landau distribution for mu = 0, c = 1
     * @param x
     * @return
     */
    Float f(Float const& x) const final
    {
        Float exponent = -(x + std::exp(-x)) / 2.0f;
        return std::exp(exponent) / std::sqrt(2.0f * F_PI);
    }

    Float max() const final
    {
        return f(0.0f);
    }
};

template<class TGenerator>
class ILandauDistribution : public LandauDistribution
{
public:
    ILandauDistribution(Float x_min_, Float x_max_) : LandauDistribution(x_min_, x_max_), seq_id(0) { }

    Float sample(AbstractImportanceBuffer<LandauDistribution, TGenerator, Float>* buffer, TGenerator* gen)
    {
        return buffer->sample1D(seq_id, gen, x_range, x_min);
    }

public:
    uint32_t seq_id;
};

class HyperbolicSecantDistribution : public Distribution1D
{
public:
    HyperbolicSecantDistribution(Float x_min_, Float x_max_) : Distribution1D(x_min_, x_max_) { }

    Float f(Float const& x) const final
    {
        auto sech = [](Float a){
            return 2.0f * std::exp(a) / (std::exp(2.0f * a) + 1);
        };
        return 0.5f * sech(F_PI_2 * x);
    }

    Float max() const final
    {
        return f(0.0f);
    }
};

template<class TGenerator>
class IHyperbolicSecantDistribution : public HyperbolicSecantDistribution
{
public:
    IHyperbolicSecantDistribution(Float x_min_, Float x_max_) : HyperbolicSecantDistribution(x_min_, x_max_), seq_id(0) { }

    Float sample(AbstractImportanceBuffer<HyperbolicSecantDistribution, TGenerator, Float>* buffer, TGenerator* gen)
    {
        return buffer->sample1D(seq_id, gen, x_range, x_min);
    }

public:
    uint32_t seq_id;
};

class GumbelDistribution : public Distribution1D
{
public:
    GumbelDistribution(Float mu_, Float beta_, Float x_min_, Float x_max_)
        : Distribution1D(x_min_, x_max_), mu(mu_), beta(beta_)  { }

    Float f(Float const& x) const final
    {
        Float z = (x - mu) / beta;
        return (1.0f / beta) * std::exp(-(z + std::exp(-z)));
    }

    Float max() const final
    {
        return f(mu);
    }

private:
    Float mu;
    Float beta;
};

template<class TGenerator>
class IGumbelDistribution : public GumbelDistribution
{
public:
    IGumbelDistribution(Float mu_, Float beta_, Float x_min_, Float x_max_)
        : GumbelDistribution(mu_, beta_, x_min_, x_max_), seq_id(0) { }

    Float sample(AbstractImportanceBuffer<GumbelDistribution, TGenerator, Float>* buffer, TGenerator* gen)
    {
        return buffer->sample1D(seq_id, gen, x_range, x_min);
    }

public:
    uint32_t seq_id;
};

class LogisticDistribution : public Distribution1D
{
public:
    LogisticDistribution(Float mu_, Float s_, Float x_min_, Float x_max_)
        : Distribution1D(x_min_, x_max_), mu(mu_), s(s_) { }

    Float f(Float const& x) const final
    {
        Float e = std::exp(-(x - mu) / s);
        return e / (s * (1.0f + e) * (1.0f + e));
    }

    Float max() const final
    {
        return f(mu);
    }

private:
    Float mu;
    Float s;
};

template<class TGenerator>
class ILogisticDistribution : public LogisticDistribution
{
public:
    ILogisticDistribution(Float mu_, Float s_, Float x_min_, Float x_max_)
        : LogisticDistribution(mu_, s_, x_min_, x_max_), seq_id(0) { }

    Float sample(AbstractImportanceBuffer<LogisticDistribution, TGenerator, Float>* buffer, TGenerator* gen)
    {
        return buffer->sample1D(seq_id, gen, x_range, x_min);
    }

public:
    uint32_t seq_id;
};

class ExponentialDistribution : public Distribution1D
{
public:
    ExponentialDistribution(Float lambda_, Float x_min_, Float x_max_)
        : Distribution1D(x_min_, x_max_), lambda(lambda_)
    {
        ASSERT(lambda_ > 0);
    }

    Float f(Float const& x) const final
    {
        return x >= 0.0f ? (lambda * std::exp(-lambda * x)) : 0.0f;
    }

    Float max() const final
    {
        return f(0.0f);
    }

private:
    Float lambda;
};

template<class TGenerator>
class IExponentialDistribution : public ExponentialDistribution
{
public:
    IExponentialDistribution(Float lambda_, Float x_min_, Float x_max_)
        : ExponentialDistribution(lambda_, x_min_, x_max_), seq_id(0) { }

    Float sample(AbstractImportanceBuffer<ExponentialDistribution, TGenerator, Float>* buffer, TGenerator* gen)
    {
        return buffer->sample1D(seq_id, gen, x_range, x_min);
    }

public:
    uint32_t seq_id;
};

class WeibullDistribution : public Distribution1D
{
public:
    WeibullDistribution(Float lambda_, Float k_, Float x_min_, Float x_max_)
        : Distribution1D(x_min_, x_max_), lambda(lambda_), k(k_)
    {
        ASSERT(lambda_ > 0);
    }

    Float f(Float const& x) const final
    {
        if (x < 0.0f)
            return 0.0f;
        return k / lambda * std::pow(x / lambda, k - 1) * std::exp(-std::pow(x / lambda, k));
    }

    Float max() const final
    {
        if (k > 1.0f)
        {
            Float mode = lambda * std::pow((k - 1.0f) / k, 1.0f / k);
            return f(mode);
        }
        if (k < 1.0f)
            return f(0.01f); // roughly guessed, value at 0 is infinity
        return f(0.0f);
    }

private:
    Float lambda;
    Float k;
};

template<class TGenerator>
class IWeibullDistribution : public WeibullDistribution
{
public:
    IWeibullDistribution(Float lambda_, Float k_, Float x_min_, Float x_max_)
        : WeibullDistribution(lambda_, k_, x_min_, x_max_), seq_id(0) { }

    Float sample(AbstractImportanceBuffer<WeibullDistribution, TGenerator, Float>* buffer, TGenerator* gen)
    {
        return buffer->sample1D(seq_id, gen, x_range, x_min);
    }

public:
    uint32_t seq_id;
};

class CauchyDistribution : public Distribution1D
{
public:
    CauchyDistribution(Float x0_, Float gamma_, Float x_min_, Float x_max_)
    : Distribution1D(x_min_, x_max_), x0(x0_), gamma(gamma_) { }

    Float f(Float const& x) const final
    {
        Float fraction = F_PI * gamma * (1.0f + std::pow((x - x0) / gamma, 2.0f));
        return 1.0f / fraction;
    }

    Float max() const final
    {
        return f(x0);
    }

private:
    Float x0;
    Float gamma;
};

template<class TGenerator>
class ICauchyDistribution : public CauchyDistribution
{
public:
    ICauchyDistribution(Float x0_, Float gamma_, Float x_min_, Float x_max_)
        : CauchyDistribution(x0_, gamma_, x_min_, x_max_), seq_id(0) { }

    Float sample(AbstractImportanceBuffer<CauchyDistribution, TGenerator, Float>* buffer, TGenerator* gen)
    {
        return buffer->sample1D(seq_id, gen, x_range, x_min);
    }

public:
    uint32_t seq_id;
};

class LaplaceDistribution : public Distribution1D
{
public:
    LaplaceDistribution(Float mu_, Float b_, Float x_min_, Float x_max_)
        : Distribution1D(x_min_, x_max_), mu(mu_), b(b_) { }

    Float f(Float const& x) const final
    {
        Float exponent = -std::abs(x - mu) / b;
        return 1.0f / (2.0f * b) * std::exp(exponent);
    }

    Float max() const final
    {
        return f(mu);
    }

private:
    Float mu;
    Float b;
};

template<class TGenerator>
class ILaplaceDistribution : public LaplaceDistribution
{
public:
    ILaplaceDistribution(Float mu_, Float b_, Float x_min_, Float x_max_)
        : LaplaceDistribution(mu_, b_, x_min_, x_max_), seq_id(0) { }

    Float sample(AbstractImportanceBuffer<LaplaceDistribution, TGenerator, Float>* buffer, TGenerator* gen)
    {
        return buffer->sample1D(seq_id, gen, x_range, x_min);
    }

private:
    uint32_t seq_id;
};

class RayleighDistribution : public Distribution1D
{
public:
    RayleighDistribution(Float sigma_, Float x_min_, Float x_max_)
        : Distribution1D(x_min_, x_max_), sigma(sigma_) { }

    Float f(Float const& x) const final
    {
        Float exponent = -x * x / (2.0f * sigma * sigma);
        return x / (sigma * sigma) * std::exp(exponent);
    }

    Float max() const final
    {
        return f(sigma);
    }

private:
    Float sigma;
};

template<class TGenerator>
class IRayleighDistribution : public RayleighDistribution
{
public:
    IRayleighDistribution(Float sigma_, Float x_min_, Float x_max_)
        : RayleighDistribution(sigma_, x_min_, x_max_), seq_id(0) { }

    Float sample(AbstractImportanceBuffer<RayleighDistribution, TGenerator, Float>* buffer, TGenerator* gen)
    {
        return buffer->sample1D(seq_id, gen, x_range, x_min);
    }

private:
    uint32_t seq_id;
};

class FrechetDistribution : public Distribution1D
{
public:
    FrechetDistribution(Float alpha_, Float s_, Float m_, Float x_min_, Float x_max_)
        : Distribution1D(x_min_, x_max_), alpha(alpha_), s(s_), m(m_) { }

    Float f(Float const& x) const final
    {
        Float term = (x - m) / s;
        Float exponent = -std::pow(term, -alpha);
        return alpha / s * std::pow(term, -1.0f - alpha) * std::exp(exponent);
    }

    Float max() const final
    {
        return f(m + s / std::pow(alpha, 1.0f / alpha));
    }

private:
    Float alpha;
    Float s;
    Float m;
};

template<class TGenerator>
class IFrechetDistribution : public FrechetDistribution
{
public:
    IFrechetDistribution(Float alpha_, Float s_, Float m_, Float x_min_, Float x_max_)
        : FrechetDistribution(alpha_, s_, m_, x_min_, x_max_), seq_id(0) { }

    Float sample(AbstractImportanceBuffer<FrechetDistribution, TGenerator, Float>* buffer, TGenerator* gen)
    {
        return buffer->sample1D(seq_id, gen, x_range, x_min);
    }

private:
    uint32_t seq_id;
};

} // namespace Importance

#endif // IMPORTANCE_MISC_DISTR_H__

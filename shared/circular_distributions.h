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

#ifndef IMPORTANCE_CIRCULAR_DISTR_H__
#define IMPORTANCE_CIRCULAR_DISTR_H__

#include <shared/distribution.h>
#include <is/ibuffer.h>

namespace Importance
{

class VonMisesDistribution : public CircularDistribution
{
public:
    VonMisesDistribution(Float mu_, Float k_) : CircularDistribution(), mu(mu_), k(k_) { }

    Float f(Float const& x) const final
    {
        return constant() * std::exp(k * std::cos(x - mu));
    }

    [[nodiscard]] Float max() const override
    {
        return f(mu);
    }

protected:
    Float constant() const
    {
        Float denominator = 2.0f * F_PI * std::cyl_bessel_i(0, k);
        return 1.0f / denominator;
    }

    Float mu;
    Float k;
};

template<class TGenerator>
class IVonMisesDistribution : public VonMisesDistribution
{
public:
    IVonMisesDistribution(Float mu_, Float k_)
        : VonMisesDistribution(mu_, k_), seq_id(0) { }

    Float sample(AbstractImportanceBuffer<VonMisesDistribution, TGenerator, Float>* buffer, TGenerator* gen)
    {
        return buffer->sampleCircle(seq_id, gen);
    }

public:
    uint32_t seq_id;
};

class WrappedCauchyDistribution : public CircularDistribution
{
public:
    WrappedCauchyDistribution(Float mu_, Float gamma_) : CircularDistribution(), mu(mu_), gamma(gamma_) { }

    Float f(Float const& x) const final
    {
        Float constant = 1.0f / (2.0f * F_PI);
        return constant * std::sinh(gamma) / (std::cosh(gamma) - std::cos(x - mu));
    }

    [[nodiscard]] Float max() const override
    {
        return f(mu);
    }

protected:
    Float mu;
    Float gamma;
};

template<class TGenerator>
class IWrappedCauchyDistribution : public WrappedCauchyDistribution
{
public:
    IWrappedCauchyDistribution(Float mu_, Float gamma_)
        : WrappedCauchyDistribution(mu_, gamma_), seq_id(0) { }

    Float sample(AbstractImportanceBuffer<WrappedCauchyDistribution, TGenerator, Float>* buffer, TGenerator* gen)
    {
        return buffer->sampleCircle(seq_id, gen);
    }

public:
    uint32_t seq_id;
};

class CardioidDistribution : public CircularDistribution
{
public:
    CardioidDistribution(Float mu_, Float ro_) : CircularDistribution(), mu(mu_), ro(ro_) { }

    Float f(Float const& x) const final
    {
        return 1.0 / (2.0f * F_PI) * (1.0f + 2.0f * ro * std::cos(x - mu));
    }

    [[nodiscard]] Float max() const override
    {
        return f(mu);
    }

protected:
    Float mu;
    Float ro;
};

template<class TGenerator>
class ICardioidDistribution : public CardioidDistribution
{
public:
    ICardioidDistribution(Float mu_, Float ro_)
        : CardioidDistribution(mu_, ro_), seq_id(0) { }

    Float sample(AbstractImportanceBuffer<CardioidDistribution, TGenerator, Float>* buffer, TGenerator* gen)
    {
        return buffer->sampleCircle(seq_id, gen);
    }

public:
    uint32_t seq_id;
};

class WrappedExponentialDistribution : public CircularDistribution
{
public:
    explicit WrappedExponentialDistribution(Float lambda_) : CircularDistribution(), lambda(lambda_) { }

    Float f(Float const& x) const final
    {
        return lambda * std::exp(-lambda * x) / (1.0f - std::exp(-2.0f * F_PI * lambda));
    }

    [[nodiscard]] Float max() const override
    {
        return f(0.0f);
    }

private:
    Float lambda;
};

template<class TGenerator>
class IWrappedExponentialDistribution : public WrappedExponentialDistribution
{
public:
    IWrappedExponentialDistribution(Float lambda_)
        : WrappedExponentialDistribution(lambda_), seq_id(0) { }

    Float sample(AbstractImportanceBuffer<WrappedExponentialDistribution, TGenerator, Float>* buffer, TGenerator* gen)
    {
        return buffer->sampleCircle(seq_id, gen);
    }

public:
    uint32_t seq_id;
};

} // namespace Importance

#endif // IMPORTANCE_CIRCULAR_DISTR_H__

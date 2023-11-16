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

#ifndef IMPORTANCE_MIXTURE_DISTR_H
#define IMPORTANCE_MIXTURE_DISTR_H

#include <shared/distribution.h>
#include <is/ibuffer.h>
#include <utility>

namespace Importance
{

class GaussianMixtureModel : public Distribution2D
{
public:
    GaussianMixtureModel(std::vector<glm::vec2> muVec_, std::vector<glm::vec2> siVec_, std::vector<Float> roVec_,
                         std::vector<Float> mixVec_, glm::vec2 min_, glm::vec2 max_)
                         : Distribution2D(min_, max_), mu(std::move(muVec_)), si(std::move(siVec_)),
                         ro(std::move(roVec_)), mix(std::move(mixVec_)) { }

    [[nodiscard]] Float f(glm::vec2 const& x) const final
    {
        Float value = 0.0f;
        for (std::size_t i = 0; i < mu.size(); i++)
            value += mix[i] * lobeValue(x, mu[i], si[i], ro[i]);

        return value;
    }

    Float max() const override
    {
        return 0.0f;
    }

private:
    [[nodiscard]] static Float lobeValue(glm::vec2 const& x, glm::vec2 lobeMu, glm::vec2 lobeSi, Float lobeRo)
    {
        Float const xt = (x.x - lobeMu.x) / lobeSi.x;
        Float const yt = (x.y - lobeMu.y) / lobeSi.y;
        Float const rot = 1.0f - lobeRo * lobeRo;
        Float const frac = 2.0f * F_PI * lobeSi.x * lobeSi.y * std::sqrt(rot);
        Float const exponent = -1.0f / (2.0f * rot) * (xt * xt + yt * yt - 2.0f * lobeRo * xt * yt);

        return 1.0f / frac * std::exp(exponent);
    }

    std::vector<glm::vec2> mu;
    std::vector<glm::vec2> si;
    std::vector<Float> ro;
    std::vector<Float> mix;
};

template<class TGenerator>
class IGaussianMixtureModel : public GaussianMixtureModel
{
public:
    IGaussianMixtureModel(std::vector<glm::vec2> muVec_, std::vector<glm::vec2> siVec_, std::vector<Float> roVec_,
                          std::vector<Float> mixVec_, glm::vec2 min_, glm::vec2 max_)
                          : GaussianMixtureModel(muVec_, siVec_, roVec_, mixVec_, min_, max_), seq_id(0) { }

    glm::vec2 sample(AbstractImportanceBuffer<GaussianMixtureModel, TGenerator, glm::vec2>* buffer, TGenerator* gen)
    {
        return buffer->sample2D(seq_id, gen, xyRange, min);
    }

private:
    uint32_t seq_id;
};

} // namespace Importance

#endif // IMPORTANCE_MIXTURE_DISTR_H

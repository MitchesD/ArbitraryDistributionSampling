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

#ifndef IMPORTANCE_IBUFFER_H__
#define IMPORTANCE_IBUFFER_H__

#include <random>
#include <utility>
#include <vector>
#include <array>
#include <deque>
#include <set>
#include <fstream>
#include <execution>
#include <algorithm>
#include <shared/config.h>
#include <shared/mapping.h>
#include <glm/glm.hpp>
#include <is/range.h>
#include <generators/lcg.h>
#include <generators/xorshift.h>

#define DETAILED_DEBUG false

namespace Importance
{

enum BufferType
{
    BUFFER_RS,
    BUFFER_RS_ADV,
    BUFFER_RS_BOX,
    BUFFER_RS_BOX_PAV,
    BUFFER_SET
};

struct InputData
{
    struct Rs { };
    struct Set { uint64_t batch_size; };

    union
    {
        Rs rs;
        Set set;
    };
};

using BufferUnit = uint32_t;

template<class TDistribution, class TGenerator, class TDomain>
class AbstractImportanceBuffer
{
public:
    AbstractImportanceBuffer() = default;
    virtual ~AbstractImportanceBuffer() = default;
    AbstractImportanceBuffer(AbstractImportanceBuffer const&) = default;
    AbstractImportanceBuffer & operator=(const AbstractImportanceBuffer&) = default;

    virtual void filterUniversal(uint32_t n, TDistribution* distribution, TGenerator* gen, InputData data) = 0;

    virtual FORCE_INLINE glm::vec3 sample(uint32_t& seq_index, TGenerator* gen)
    {
        BufferUnit skip = chain[seq_index];
        if (++seq_index >= chain.size())
        {
            gen->reset();
            seq_index = 0;
        }

        gen->skipN(2 * skip);
        glm::vec2 sample = gen->next2D();
        glm::vec3 const sphericalSample = uniformSphereSample(sample);

        return glm::normalize(sphericalSample);
    }

    virtual FORCE_INLINE Float sample1D(uint32_t& seq_index, TGenerator* gen, Float range, Float min)
    {
        BufferUnit skip = chain[seq_index];
        if (++seq_index >= chain.size())
        {
            gen->reset();
            seq_index = 0;
        }

        gen->skipN(skip);
        Float sample = gen->next1D();
        return sample * range + min;
    }

    virtual FORCE_INLINE glm::vec2 sample2D(uint32_t& seq_index, TGenerator* gen, glm::vec2 range, glm::vec2 min)
    {
        BufferUnit skip = chain[seq_index];
        if (++seq_index >= chain.size())
        {
            gen->reset();
            seq_index = 0;
        }

        gen->skipN(2 * skip);
        glm::vec2 u = gen->next2D();
        return { u.x * range.x + min.x, u.y * range.y + min.y };
    }

    virtual FORCE_INLINE Float sampleCircle(uint32_t& seq_index, TGenerator* gen)
    {
        BufferUnit skip = chain[seq_index];
        if (++seq_index >= chain.size())
        {
            gen->reset();
            seq_index = 0;
        }

        gen->skipN(skip);
        return 2.0f * F_PI * gen->next1D();
    }

    uint64_t evalTotalSkips() const
    {
        return std::accumulate(chain.begin(), chain.end(), 0);
    }

    virtual uint64_t getSkipsUpToIndex(uint32_t seq_index) const
    {
        uint64_t acc = 0;
        for (uint64_t i = 0; i < seq_index; i++)
            acc += chain[i];
        return acc;
    }

    std::vector<BufferUnit> const& getChain() { return chain; }
    void setChain(std::vector<BufferUnit> ch) { chain = std::move(ch); }

protected:
    std::vector<BufferUnit> chain;
};

template<class TDistribution, class TGenerator, class TDomain>
class RSBuffer : public AbstractImportanceBuffer<TDistribution, TGenerator, TDomain>
{
public:
    explicit RSBuffer() : AbstractImportanceBuffer<TDistribution, TGenerator, TDomain>() { }

    using IData = InputData;

    void filterUniversal(uint32_t n, TDistribution* distribution, TGenerator* gen, IData /*data*/) override
    {
        this->chain.clear();
        this->chain.reserve(n);

        uint32_t currentStep = 0;
        uint32_t totalRejected = 0;
        uint32_t total = 0;
        uint32_t maxRejected = 0;
        Float avgRejected = 0.0f;

        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<Float> uni_dis;

        Float d_max = distribution->max();
        while (this->chain.size() < n)
        {
            total++;
            Float const r1 = uni_dis(generator);
            TDomain sample = distribution->domainSample(gen);

            Float value = distribution->f(sample);
            if (r1 * d_max > value)
            {
                currentStep++;
                ASSERT(currentStep < (std::numeric_limits<BufferUnit>::max() - 10u));
                totalRejected++;
                continue;
            }

            avgRejected += (Float)currentStep;
            maxRejected = std::max(maxRejected, currentStep);
            this->chain.emplace_back(static_cast<BufferUnit>(currentStep));
            currentStep = 0;
        }

#if DETAILED_DEBUG
        std::cout << "Total rejected: " << totalRejected << " ("<< (float(totalRejected) / float(total) * 100.0f) << "%) out of " << total << std::endl;
        std::cout << "Average skip step: " << avgRejected / (Float)n << std::endl;
        std::cout << "Max. skip step: " << maxRejected << std::endl;
        std::cout << "Total sequence skips: " << this->evalTotalSkips() << std::endl;
#endif
    }
};

template<class TDistribution, class TGenerator, class TDomain>
class RSAdvancedBuffer : public AbstractImportanceBuffer<TDistribution, TGenerator, TDomain>
{
public:
    explicit RSAdvancedBuffer(Float errorPct_ = 0.15f) : AbstractImportanceBuffer<TDistribution, TGenerator, TDomain>(),
        errorPct(errorPct_) { }

    using IData = InputData;

    void filterUniversal(uint32_t n, TDistribution* distribution, TGenerator* gen, IData /*data*/) override
    {
        this->chain.clear();
        this->chain.reserve(n);

        uint32_t currentStep = 0;
        uint32_t totalRejected = 0;
        uint32_t total = 0;
        uint32_t maxRejected = 0;
        Float avgRejected = 0.0f;

        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<Float> uni_dis;

        Float d_max = distribution->max();
        while (this->chain.size() < n)
        {
            total++;
            Float const r1 = uni_dis(generator);
            TDomain sample = distribution->domainSample(gen);

            Float value = distribution->f(sample);
            Float const epsilon = errorPct * r1 * d_max;
            if (r1 * d_max - epsilon > value)
            {
                currentStep++;
                ASSERT(currentStep < (std::numeric_limits<BufferUnit>::max() - 10u));
                totalRejected++;
                continue;
            }

            avgRejected += (Float)currentStep;
            maxRejected = std::max(maxRejected, currentStep);
            this->chain.emplace_back(static_cast<BufferUnit>(currentStep));
            currentStep = 0;
        }

#if DETAILED_DEBUG
        std::cout << "Total rejected: " << totalRejected << " ("<< (float(totalRejected) / float(total) * 100.0f) << "%) out of " << total << std::endl;
        std::cout << "Average skip step: " << avgRejected / (Float)n << std::endl;
        std::cout << "Max. skip step: " << maxRejected << std::endl;
        std::cout << "Total sequence skips: " << this->evalTotalSkips() << std::endl;
#endif
    }

private:
    Float errorPct;
};

template<class TDistribution, class TGenerator, class TDomain>
class SetBuffer : public AbstractImportanceBuffer<TDistribution, TGenerator, TDomain>
{
private:
    struct BatchSample
    {
        BatchSample(TDomain sample_, Float val_) : sample(sample_), value(val_) { }

        TDomain sample;
        Float value;

        bool operator<(const BatchSample& f) const
        {
            return value < f.value;
        }
    };

public:
    SetBuffer() : AbstractImportanceBuffer<TDistribution, TGenerator, TDomain>() { }

    using IData = InputData;

    void filterUniversal(uint32_t n, TDistribution* distribution, TGenerator* gen, IData data) override
    {
        this->chain.clear();
        this->chain.reserve(n);

        uint32_t currentStep = 0;
        uint32_t total = 0;
        uint32_t maxRejected = 0;
        Float avgRejected = 0.0f;

        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<Float> uni_dis;

        // 1. create N samples
        // 2. each sample should have a weight represented by function value
        // 3. sort the samples with weight in ascending order
        // 4. create normalized cumulative weight sum
        // 5. sample r ~ [0, 1), pick sample according to cumulative sum, mark it as accepted, repeat N steps, discard samples taken more times
        // 6. reconstruct skipping sequence
        while (this->chain.size() < n)
        {
            // 1, 2
            Float sum = 0.0f;
            std::vector<BatchSample> set;
            for (uint32_t j = 0; j < data.set.batch_size; j++)
            {
                TDomain sample = distribution->domainSample(gen);
                Float value = distribution->f(sample);
                set.emplace_back(sample, value);
                sum += value;
                total++;
            }

            // 3
            auto sortedIndexes = sortIndexes(set);
            // 4
            std::vector<Float> cumulativeSum;
            Float localSum = 0.0f;
            for (auto const& item : sortedIndexes)
            {
                localSum += set[item].value;
                cumulativeSum.push_back(localSum / sum);
            }

            // 5
            std::set<uint32_t> acceptedIndexes;
            for (uint32_t j = 0; j < data.set.batch_size && (this->chain.size() + acceptedIndexes.size()) < n; j++)
            {
                Float const r1 = uni_dis(generator);
                // binary search for index where r1 < cumulativeSum[index]
                auto itr = std::lower_bound(cumulativeSum.begin(), cumulativeSum.end(), r1) ;
                auto index = std::distance(cumulativeSum.begin(), itr);
                acceptedIndexes.insert(sortedIndexes[index]);
            }

            // 6
            uint32_t prevIndex = 0;
            uint32_t skipStep = 0;
            for (auto const& index : acceptedIndexes)
            {
                uint32_t step = currentStep + index - prevIndex;
                currentStep = 0;
                prevIndex = index + 1;
                skipStep = index;
                this->chain.emplace_back(static_cast<BufferUnit>(step));
                avgRejected += (Float)step;
                maxRejected = std::max(maxRejected, step);
            }

            // update current step, to next iteration
            currentStep = data.set.batch_size - skipStep;
            std::cout << std::endl;
            std::cout << "current step: " << currentStep << std::endl;
            std::cout << "Accepted samples: " << acceptedIndexes.size() << " (" << Float(acceptedIndexes.size()) / Float(data.set.batch_size) * 100.0f << "%)" << std::endl;
        }

#if DETAILED_DEBUG
        std::cout << "Total accepted: " << this->chain.size() << " ("<< (float(this->chain.size()) / float(total) * 100.0f) << "%) out of " << total << std::endl;
        std::cout << "Total rejected in pct: " << (1.0f - float(this->chain.size()) / float(total)) * 100.0f << "%" << std::endl;
        std::cout << "Average skip step: " << avgRejected / (Float)n << std::endl;
        std::cout << "Max. skip step: " << maxRejected << std::endl;
        std::cout << "Total sequence skips: " << this->evalTotalSkips() << std::endl;
#endif
    }

private:
    template<typename C, typename T = typename C::value_type>
    std::vector<size_t> sortIndexes(C const& v)
    {
        // initialize original index locations
        std::vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        // sort indexes based on comparing values in v
        // using std::stable_sort instead of std::sort
        // to avoid unnecessary index re-orderings
        // when v contains elements of equal values
        std::stable_sort(std::execution::par_unseq, idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
            return v[i1] < v[i2];
        });

        return idx;
    }
};

BufferUnit constexpr boxBits = 12u;
BufferUnit constexpr boxMask = (1u << boxBits) - 1u;
BufferUnit constexpr strideBits = 20u;
BufferUnit constexpr bitMask = BufferUnit(~(boxMask << strideBits));
BufferUnit constexpr idBitMask = BufferUnit(boxMask << strideBits);

/**
 * This class implements the RS buffer with grid partitioning.
 * @tparam TDistribution
 * @tparam TGenerator
 * @tparam TDomain
 * @tparam TGrid
 */
template<class TDistribution, class TGenerator, class TDomain, class TGrid>
class RSWrappedAdvancedBuffer : public AbstractImportanceBuffer<TDistribution, TGenerator, TDomain>
{
public:
    template<class... Args>
    explicit RSWrappedAdvancedBuffer(Args... args) : AbstractImportanceBuffer<TDistribution, TGenerator, TDomain>(),
                                                         grid(new TGrid(args...)) { }

    ~RSWrappedAdvancedBuffer()
    {
        delete grid;
    }

    void filterUniversal(uint32_t n, TDistribution* distribution, TGenerator* gen, InputData /*data*/) override
    {
        uint32_t totalRejected = 0;
        uint32_t total = 0;
        uint32_t totalAccepted = 0;
        uint32_t maxRejected = 0;
        Float avgRejected = 0.0f;

        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<Float> uni_dis;

        // 1. setup grid
        // 2. eval maximal values on intervals - at each grid node
        grid->prepare(distribution);
        uint32_t nodeCount = grid->getNodesCount();
        std::vector<uint32_t> stats;
        stats.resize(nodeCount, 0);

        // 3. perform RS in the following manner:
        while (totalAccepted < n)
        {
            // 3a. select grid node
            auto grid_pair = grid->sampleNode(uni_dis(generator));
            auto node = grid_pair.second;
            auto node_id = node.id;
            uint32_t currentStep = 0;
            stats[node_id]++;

            // ID must be representable with 6-bits by default
            ASSERT(node_id < (1 << boxBits));
            BufferUnit shifted_id = node_id << strideBits;
            while (true)
            {
                // 3b. sample value in range, reject or accept, remember current skip step in grid node
                total++;
                Float const r1 = uni_dis(generator);
                TDomain sample = distribution->domainSample(gen, grid_pair.first);
                Float value = distribution->f(sample);

                if (r1 * node.localMax > value)
                {
                    currentStep++;
                    ASSERT(currentStep < (std::numeric_limits<BufferUnit>::max() - 10u));
                    totalRejected++;
                    continue;
                }

                // 3c. if accepted, store skip step in chain array with current grid node ID
                avgRejected += (Float) currentStep;
                maxRejected = std::max(maxRejected, currentStep);
                // currentStep must be representable with 10 bits
                ASSERT_MESSAGE(currentStep < (1 << strideBits), currentStep);
                currentStep |= shifted_id;
                this->chain.emplace_back(static_cast<BufferUnit>(currentStep));
                totalAccepted++;
                break;
            }
        }

#if DETAILED_DEBUG
        std::cout << "Total rejected: " << totalRejected << " ("<< (float(totalRejected) / float(total) * 100.0f) << "%) out of " << total << std::endl;
        std::cout << "Average skip step: " << avgRejected / (Float)n << std::endl;
        std::cout << "Max. skip step: " << maxRejected << std::endl;
        std::cout << "Total sequence skips: " << this->evalTotalSkips() << std::endl;

        int i = 0;
        for (auto const& item : stats)
            std::cout << i++ << " - " << item << " (" << Float(item) / Float(totalAccepted) * 100.0f << "%)" << std::endl;
#endif
    }

    FORCE_INLINE glm::vec3 sample(uint32_t& seq_index, TGenerator* gen) override
    {
        BufferUnit holder = this->chain[seq_index];
        BufferUnit const skip = (holder & bitMask);

        BufferUnit const nodeId = ((holder & idBitMask) >> strideBits);
        GridRange range = grid->getNodeGridRange(nodeId);
        RangeVector r = std::get<RangeVector>(range);

        if (++seq_index >= this->chain.size())
        {
            gen->reset();
            seq_index = 0;
        }

        gen->skipN(2 * skip);
        glm::vec2 sample = gen->next2D();
        Float theta = (r.max.y - r.min.y) * sample[0] + r.min.y;
        ASSERT_MESSAGE(theta >= r.min.y && theta <= r.max.y, theta);
        Float cosTheta = std::cos(theta);
        Float sinTheta = std::sin(theta);
        Float phi = sample[1] * (r.max.x - r.min.x) + r.min.x;
        ASSERT_MESSAGE(phi >= r.min.x && phi <= r.max.x, phi);

        return { sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta };
    }

    FORCE_INLINE Float sample1D(uint32_t& seq_index, TGenerator* gen, Float /*range*/, Float /*min*/) override
    {
        BufferUnit holder = this->chain[seq_index];

        BufferUnit const skip = (holder & bitMask);

        // map nodeId to new "range" and "min"
        BufferUnit const nodeId = ((holder & idBitMask) >> strideBits);
        GridRange gr = grid->getNodeGridRange(nodeId);
        RangeScalar r = std::get<RangeScalar>(gr);

        if (++seq_index >= this->chain.size())
        {
            gen->reset();
            seq_index = 0;
        }

        gen->skipN(skip);
        Float sample = gen->next1D();
        return sample * (r.max - r.min) + r.min;
    }

    virtual FORCE_INLINE glm::vec2 sample2D(uint32_t& seq_index, TGenerator* gen, glm::vec2 /*range*/, glm::vec2 /*min*/) override
    {
        BufferUnit holder = this->chain[seq_index];

        BufferUnit const skip = (holder & bitMask);

        // map nodeId to new "range" and "min"
        BufferUnit const nodeId = ((holder & idBitMask) >> strideBits);
        GridRange gr = grid->getNodeGridRange(nodeId);
        RangeVector r = std::get<RangeVector>(gr);

        if (++seq_index >= this->chain.size())
        {
            gen->reset();
            seq_index = 0;
        }

        gen->skipN(2 * skip);
        glm::vec2 u = gen->next2D();
        return { u.x * (r.max.x - r.min.x) + r.min.x, u.y * (r.max.y - r.min.y) + r.min.y };
    }

    FORCE_INLINE Float sampleCircle(uint32_t& seq_index, TGenerator* gen) override
    {
        BufferUnit holder = this->chain[seq_index];

        BufferUnit const skip = (holder & bitMask);

        // map nodeId to new "range" and "min"
        BufferUnit const nodeId = ((holder & idBitMask) >> strideBits);
        GridRange gr = grid->getNodeGridRange(nodeId);
        RangeScalar range = std::get<RangeScalar>(gr);

        if (++seq_index >= this->chain.size())
        {
            gen->reset();
            seq_index = 0;
        }

        gen->skipN(skip);
        return gen->next1D() * (range.max - range.min) + range.min;
    }

    uint64_t getSkipsUpToIndex(uint32_t seq_index) const override
    {
        uint64_t acc = 0;
        for (uint64_t i = 0; i < seq_index; i++)
            acc += this->chain[i] & bitMask;
        return acc;
    }

    TGrid* getGrid() const
    {
        return grid;
    }

private:
    TGrid* grid;
};

/**
 * @brief This class implements RS buffer with grid partitioning and PAV algorithm for grid node sampling.
 * @tparam TDistribution
 * @tparam TGenerator XorShift generator is NOT supported by this method
 * @tparam TDomain
 * @tparam TGrid
 */
template<class TDistribution, class TGenerator, class TDomain, class TGrid>
class RSWrappedGridBuffer : public AbstractImportanceBuffer<TDistribution, TGenerator, TDomain>
{
public:
    static constexpr Float INV_UINT31_MAX = 1.0f / Float(0x7FFFFFFF);

    template<class... Args>
    explicit RSWrappedGridBuffer(Args... args) : AbstractImportanceBuffer<TDistribution, TGenerator, TDomain>(),
                                                         grid(new TGrid(args...)) { }

    ~RSWrappedGridBuffer()
    {
        delete grid;
    }

    void filterUniversal(uint32_t n, TDistribution* distribution, TGenerator* gen, InputData /*data*/) override
    {
        totalRejected = 0;
        total = 0;
        totalAccepted = 0;
        maxRejected = 0;
        avgRejected = 0.0f;

        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<Float> uni_dis;

        // 1. setup grid
        // 2. eval maximal values on intervals - at each grid node
        grid->setHeuristic(G_HEURISTIC_MAX);
        grid->prepare(distribution);
        uint32_t nodeCount = grid->getNodesCount();
        stats.clear();
        stats.resize(nodeCount, 0);
        uint32_t currentStep = 0;

        // 3. perform RS in the following manner:
        while (totalAccepted < n)
        {
            total++;
            Float const r1 = uni_dis(generator);

            auto [sample, localMax] = generateSample(distribution, gen);

            Float value = distribution->f(sample);
            if (r1 * localMax > value)
            {
                currentStep++;
                ASSERT(currentStep < (std::numeric_limits<BufferUnit>::max() - 10u));
                totalRejected++;
                continue;
            }

            // 3c. if accepted, store skip step in chain array with current grid node ID
            avgRejected += (Float) currentStep;
            maxRejected = std::max(maxRejected, currentStep);
            ASSERT(currentStep < (std::numeric_limits<BufferUnit>::max() - 10u));
            this->chain.emplace_back(static_cast<BufferUnit>(currentStep));
            totalAccepted++;
            currentStep = 0;
        }

#if DETAILED_DEBUG
        std::cout << "Total rejected: " << totalRejected << " ("<< (float(totalRejected) / float(total) * 100.0f) << "%) out of " << total << std::endl;
        std::cout << "Average skip step: " << avgRejected / (Float)n << std::endl;
        std::cout << "Max. skip step: " << maxRejected << std::endl;
        std::cout << "Total sequence skips: " << this->evalTotalSkips() << std::endl;

        int i = 0;
        for (auto const& item : stats)
            std::cout << i++ << " - " << item << " (" << Float(item) / Float(total) * 100.0f << "%)" << std::endl;
#endif
    }

    FORCE_INLINE glm::vec3 sample(uint32_t& seq_index, TGenerator* gen) override
    {
        BufferUnit skip = this->chain[seq_index];

        if (++seq_index >= this->chain.size())
        {
            gen->reset();
            seq_index = 0;
        }

        gen->skipN(2 * skip);
        auto [top, bottom] = gen->next2DInt();
        Float const left = ((top & 0x7FFFFFFF00000000u) >> 32u) * INV_UINT31_MAX;
        Float const right = (top & 0xFFFFFFFFu) * INV_UNSIGNED_RAND_MAX;

        auto grid_pair = grid->sampleNode(left);
        RangeVector r = std::get<RangeVector>(grid_pair.first);

        Float theta = (r.max.y - r.min.y) * right + r.min.y;
        ASSERT_MESSAGE(!(theta < r.min.y && theta > r.max.y), theta << " " << r.min.y << " " << r.max.y);
        Float cosTheta = std::cos(theta);
        Float sinTheta = std::sin(theta);
        Float phi = (bottom * INV_UINT63_MAX) * (r.max.x - r.min.x) + r.min.x;
        ASSERT_MESSAGE(phi >= r.min.x && phi <= r.max.x, phi);

        return { sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta };
    }

    FORCE_INLINE Float sample1D(uint32_t& seq_index, TGenerator* gen, Float /*range*/, Float /*min*/) override
    {
        BufferUnit skip = this->chain[seq_index];

        if (++seq_index >= this->chain.size())
        {
            gen->reset();
            seq_index = 0;
        }

        gen->skipN(skip);
        uint64_t sample = gen->next1DInt();
        Float const left = ((sample & 0x7FFFFFFF00000000u) >> 32u) * INV_UINT31_MAX;
        Float const right = (sample & 0xFFFFFFFFu) * INV_UNSIGNED_RAND_MAX;

        auto grid_pair = grid->sampleNode(left);
        auto range = grid_pair.first;
        RangeScalar r = std::get<RangeScalar>(range);

        return right * (r.max - r.min) + r.min;
    }

    virtual FORCE_INLINE glm::vec2 sample2D(uint32_t& seq_index, TGenerator* gen, glm::vec2 /*range*/, glm::vec2 /*min*/) override
    {
        BufferUnit skip = this->chain[seq_index];

        if (++seq_index >= this->chain.size())
        {
            gen->reset();
            seq_index = 0;
        }
        gen->skipN(2 * skip);
        auto [top, bottom] = gen->next2DInt();
        Float const left = ((top & 0x7FFFFFFF00000000u) >> 32u) * INV_UINT31_MAX;
        Float const right = (top & 0xFFFFFFFFu) * INV_UNSIGNED_RAND_MAX;

        auto grid_pair = grid->sampleNode(left);
        RangeVector r = std::get<RangeVector>(grid_pair.first);

        return { right * (r.max.x - r.min.x) + r.min.x, (bottom * INV_UINT63_MAX) * (r.max.y - r.min.y) + r.min.y };
    }

    FORCE_INLINE Float sampleCircle(uint32_t& seq_index, TGenerator* gen) override
    {
        BufferUnit skip = this->chain[seq_index];
        if (++seq_index >= this->chain.size())
        {
            gen->reset();
            seq_index = 0;
        }

        gen->skipN(skip);
        uint64_t sample = gen->next1DInt();
        Float const left = ((sample & 0x7FFFFFFF00000000u) >> 32u) * INV_UINT31_MAX;
        Float const right = (sample & 0xFFFFFFFFu) * INV_UNSIGNED_RAND_MAX;

        auto grid_pair = grid->sampleNode(left);
        auto range = grid_pair.first;
        RangeScalar r = std::get<RangeScalar>(range);

        return right * (r.max - r.min) + r.min;
    }

    TGrid* getGrid() const
    {
        return grid;
    }

private:
    std::tuple<TDomain, Float> generateSample(TDistribution* distribution, TGenerator* gen)
    {
        if constexpr (std::is_same_v<TDomain, Float>)
        {
            // 3a. select grid node with upper 32-bits // TODO: LCG works with 64-bits and XORShift with 32-bits
            uint64_t u = gen->next1DInt();
            Float const left = ((u & 0x7FFFFFFF00000000u) >> 32u) * INV_UINT31_MAX;
            Float const right = (u & 0xFFFFFFFFu) * INV_UNSIGNED_RAND_MAX;

            auto grid_pair = grid->sampleNode(left);
            auto node = grid_pair.second;
            stats[node.id]++;

            // 3b. sample value in range, reject or accept, remember current skip step in grid node
            return std::make_tuple(distribution->domainSample(right, grid_pair.first), node.localMax);
        }
        else if constexpr (std::is_same_v<TDomain, glm::vec3> || std::is_same_v<TDomain, glm::vec2>)
        {
            // 3a. select grid node with upper 32-bits of top element // TODO: LCG works with 64-bits and XORShift with 32-bits
            auto [top, bottom] = gen->next2DInt();
            Float const left = ((top & 0x7FFFFFFF00000000u) >> 32u) * INV_UINT31_MAX;
            Float const right = (top & 0xFFFFFFFFu) * INV_UNSIGNED_RAND_MAX;

            auto grid_pair = grid->sampleNode(left);
            auto node = grid_pair.second;

            stats[node.id]++;

            // 3b. sample value in range, reject or accept, remember current skip step in grid node
            return std::make_tuple(distribution->domainSample(glm::vec2(right, bottom * INV_UINT63_MAX),
                                                              grid_pair.first), node.localMax);
        }
        else
            ASSERT_MESSAGE("", "Unreachable code");
    }

    TGrid* grid;

    uint32_t totalRejected = 0;
    uint32_t total = 0;
    uint32_t totalAccepted = 0;
    uint32_t maxRejected = 0;
    Float avgRejected = 0.0f;

    std::vector<uint32_t> stats;
};

} // namespace Importance

#endif // IMPORTANCE_IBUFFER_H__

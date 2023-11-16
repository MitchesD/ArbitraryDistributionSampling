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

#ifndef IMPORTANCE_IVECTOR_H__
#define IMPORTANCE_IVECTOR_H__

#include <random>
#include <is/storage.h>
#include <shared/config.h>
#include <shared/fast_float.h>

namespace Importance
{

template<typename T, std::size_t N>
class StaticVector : public AlignedStorage<LinkedList<T>, N>
{
public:
    explicit StaticVector(std::vector<T> const& _data) : AlignedStorage<LinkedList<T>, N>(), samples(N)
    {
        prepare(_data);
    }

    StaticVector() : AlignedStorage<LinkedList<T>, N>(), samples(N) { }

    ~StaticVector()
    {
        clear();
    }

    void setData(std::vector<T> const& _data)
    {
        prepare(_data);
    }

    // weight pick
    T sample(Float r1);

    // forbid modifying vector components
    T& operator[](uint16_t i) = delete;

    void clear()
    {
        for (uint32_t i = 0; i < samples; i++)
            this->GetAlignedDataRef(i).Clear();
    }

private:
    // offline pre-processing
    void prepare(std::vector<T> const& values);

    uint32_t samples;
};

template<typename T, std::size_t N>
void StaticVector<T, N>::prepare(std::vector<T> const& values)
{
#ifdef DEBUG
    // assuming already normalized weights as input
    auto acc_func = [] (const Float acc, auto const& item) {
        return acc + item;
    };

    auto float_close = [](Float a, Float b) {
        return std::abs(a - b) < 1e-5;
    };
    ASSERT_MESSAGE(float_close(std::accumulate(values.begin(), values.end(), 0.0, acc_func), 1.0),
           "prepare_weighted_array: weights are not normalized, the sum is: "
               << std::accumulate(values.begin(), values.end(), 0.0, acc_func));
#endif

    Float sum = 0.0f;
    Float acc = 0.0f;
    Float increment = 1.0f / Float(samples);
    uint32_t current_index = 0;
    std::vector<Float> cum_sum;
    Float prevSplit = 0.0f;
    Float split = 0.0f;

    Float tmp = 0.0f;
    for (auto const& item : values)
    {
        tmp += item;
        cum_sum.push_back(tmp);
    }

    // fill aligned storage with empty lists
    for (uint32_t i = 0; i < samples; i++)
        this->emplace_back(LinkedList<T>());

    // fill linked lists with values
    for (uint32_t i = 0; i < samples; )
    {
        if (current_index >= values.size())
            break;

        // [acc, acc + inc)
        Float acc_min = acc;
        Float acc_max = acc_min + increment;
        Float current_value = values[current_index];
        split = cum_sum[current_index] >= acc_max ? acc_max : cum_sum[current_index];
        Float weight = split - prevSplit;

        // 1. case - value lies over the whole interval
        if (sum + current_value >= acc_max)
        {
            acc += increment;
            this->GetAlignedDataRef(i++).Append(values[current_index], weight);
        }
        // 2. case - value ends in interval and another one starts in interval
        else
        {
            this->GetAlignedDataRef(i).Append(values[current_index], weight);
            sum += current_value;
            current_index++;
        }

        prevSplit = split;
    }
}

template<typename T, std::size_t N>
T StaticVector<T, N>::sample(Float r1)
{
    uint32_t r1i = r1 * (samples);
    auto class_array = this->GetAlignedData(r1i);

    // if vector has just one item, return in O(1), otherwise continue with O(N)
    if (class_array.HasOneItem())
        return class_array.head->data;

    Float acc = 0.0;
    Float const range = 1.0f / Float(samples);
    Float rng = r1 - (r1i * range);

    LinkedListNode<T>* itr = class_array.head;
    while (itr)
    {
        acc += itr->weight;
        if (acc >= rng)
            return itr->data;

        itr = itr->next;
    }

    // should never happen
    return class_array.head->data; // TODO: fix this case
}

template<typename T>
class DynamicVector
{
public:
    explicit DynamicVector(std::vector<T> const& _data) : data(_data), partial_sums(nullptr), ready(false)
    {
        ComputePartialSums();
        ComputeIntervals();
    }

    DynamicVector() : partial_sums(nullptr) { ready = false; }

    ~DynamicVector()
    {
        delete [] partial_sums;
    }

    // 1. method uses prefix sum
    T SampleM1(float r1);

    // 2. method uses intervals
    T SampleM2(float r1);

    // update the whole vector if needed, as it is dynamic structure
    void UpdateVector(std::vector<T> const& _data)
    {
        data = _data;
        ready = false;
    }

    void Invalidate()
    {
        ready = false;
    }

    // get/set single component of vector
    const T& operator[](uint16_t i) const { return data[i]; }
    T& operator[](uint16_t i) { return data[i]; }

private:
    void ComputePartialSums()
    {
        delete [] partial_sums;

        partial_sums = new float[data.size()];
        std::copy(data.begin(), data.end(), partial_sums);
        avx2_sum(partial_sums, data.size());
        ready = true;
    }

    void ComputeIntervals()
    {
        intervals.clear();
        intervals.resize(data.size());
        float sum = 0.0f;
        float sump = 0.0f;
        uint16_t i = 0;
        for (auto const& item : data)
        {
            sum += item;
            intervals[i++] = {sump, sum};
            sump += item;
        }
        ready = true;
    }

    std::vector<T> data;

    float* partial_sums;
    std::vector<std::pair<float, float>> intervals;
    bool ready;
};

template<typename T>
T DynamicVector<T>::SampleM1(float r1)
{
    // fill vectors with partial sums aka pre-scan, if not already done
    if (!ready)
        ComputePartialSums();

    int32_t left = 0;
    int32_t right = data.size() - 1;
    while (left <= right)
    {
        int32_t mid = left + (right - left) / 2;
        auto value = partial_sums[mid];

        if (r1 >= value)
        {
            // go to right
            left = mid + 1;
        }
        else
        {
            // if bigger than prev, return mid
            float prev = mid - 1 < 0 ? 0.0f : partial_sums[mid - 1];
            if (r1 >= prev)
                return data[mid];

            // go to left
            right = mid - 1;
        }
    }

    // should never happen
    ASSERT_MESSAGE(false, "weight_pick_dynamic: reached dead code");
    return T(-1);
}

template<typename T>
T DynamicVector<T>::SampleM2(float r1)
{
    if (!ready)
        ComputeIntervals();

    uint32_t left = 0;
    uint32_t right = intervals.size() - 1;
    while (left <= right)
    {
        uint32_t mid = left + (right - left) / 2;
        auto pair_value = intervals[mid];

        if (r1 >= pair_value.second)
        {
            // go to right
            left = mid + 1;
        }
        else if (r1 < pair_value.first)
        {
            // go to left
            right = mid - 1;
        }
        else
            return data[mid];
    }

    // should never happen
    ASSERT_MESSAGE(false, "weight_pick_dynamic: reached dead code");
    return T();
}

} // namespace Importance

#endif // IMPORTANCE_IVECTOR_H__

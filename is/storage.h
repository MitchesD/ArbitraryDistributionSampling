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

#ifndef IMPORTANCE_LINKED_LIST_H__
#define IMPORTANCE_LINKED_LIST_H__

#include <algorithm>
#include <type_traits>
#include <iostream>

namespace Importance
{

template<class T>
struct LinkedListNode
{
    T data;
    LinkedListNode* next;
    // weight in current cell
    float weight;
};

template<class T>
class LinkedList
{
public:
    LinkedList() : head(nullptr), tail(nullptr) { }

    void Clear()
    {
        LinkedListNode<T>* current = head;
        LinkedListNode<T>* next = nullptr;

        while (current != nullptr)
        {
            next = current->next;
            delete current;
            current = next;
        }

        head = nullptr;
        tail = nullptr;
    }

    void Append(T const& data, float weight)
    {
        auto item = new LinkedListNode<T>();
        item->data = data;
        item->next = nullptr;
        item->weight = weight;

        // initialize head
        if (!head)
            head = item;
        // update tail _next_ and set
        if (tail)
            tail->next = item;
        tail = item;
    }

    [[nodiscard]] bool HasOneItem() const
    {
        return head == tail;
    }

    // _T_ has to be convertible to float
    [[nodiscard]] float SumNodes() const
    {
        float sum = 0.0f;
        auto itr = head;
        while (itr)
        {
            sum += itr->data;
            itr = itr->next;
        }

        return sum;
    }

    LinkedListNode<T>* head;
    LinkedListNode<T>* tail;
};

template<class T, std::size_t N>
class AlignedStorage
{
protected:
    // properly aligned uninitialized storage for N T's
    //alignas(16) std::array<std::byte[sizeof(T)], N> data;
    typename std::aligned_storage<sizeof(T), alignof(T)>::type data[N];
    std::size_t size = 0;

public:
    // create an object in aligned storage
    template<typename ...Args> void emplace_back(Args&&... args)
    {
        if (size >= N)
            throw std::bad_alloc{};

        // construct value in memory of aligned storage
        // using inplace operator new
        new(&data[size]) T(std::forward<Args>(args)...);
        ++size;
    }

    // read-only access an object in aligned storage
    const T& operator[](std::size_t pos) const
    {
        return *std::launder(reinterpret_cast<const T*>(&data[pos]));
    }

    // read-only access an object in aligned storage
    T const& GetAlignedData(std::size_t pos) const
    {
        return *std::launder(reinterpret_cast<const T*>(&data[pos]));
    }

    // read-write access an object in aligned storage
    T& GetAlignedDataRef(std::size_t pos)
    {
        return *std::launder(reinterpret_cast<T*>(&data[pos]));
    }

    // delete objects from aligned storage
    ~AlignedStorage()
    {
        for (std::size_t pos = 0; pos < size; ++pos)
            std::launder(reinterpret_cast<T*>(&data[pos]))->~T();
    }
};

} // namespace Importance

#endif // IMPORTANCE_LINKED_LIST_H__

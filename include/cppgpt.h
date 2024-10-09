#ifndef INC_CPPGPT_H_
#define INC_CPPGPT_H_
#include "gguf.h"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <istream>
#include <memory>
#include <type_traits>
#include <utility>
#include <string>
#include <functional>

// new/delete
void* operator new(std::size_t size);
void* operator new(std::size_t size, std::align_val_t alignment);
void* operator new(std::size_t size, const std::nothrow_t&) noexcept;
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept;
// void* operator new(std::size_t size, void* ptr) noexcept;
void operator delete(void* ptr) noexcept;
void operator delete(void* ptr, std::size_t size) noexcept;
void operator delete(void* ptr, std::align_val_t alignment) noexcept;
void operator delete(void* ptr, std::size_t size, std::align_val_t alignment) noexcept;
void operator delete(void* ptr, const std::nothrow_t&) noexcept;
void operator delete(void* ptr, std::align_val_t alignment, const std::nothrow_t&) noexcept;
// void operator delete(void* ptr, void*) noexcept;

void* operator new[](std::size_t size);
void* operator new[](std::size_t size, std::align_val_t alignment);
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept;
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept;
// void* operator new[](std::size_t size, void* ptr) noexcept;
void operator delete[](void* ptr) noexcept;
void operator delete[](void* ptr, std::size_t size) noexcept;
void operator delete[](void* ptr, std::align_val_t alignment) noexcept;
void operator delete[](void* ptr, std::size_t size, std::align_val_t alignment) noexcept;
void operator delete[](void* ptr, const std::nothrow_t&) noexcept;
void operator delete[](void* ptr, std::align_val_t alignment, const std::nothrow_t&) noexcept;
// void operator delete[](void* ptr, void*) noexcept;

namespace gguf
{
class GGUF;
}

namespace cppgpt
{
using s8 = int8_t;
using s16 = int16_t;
using s32 = int32_t;
using s64 = int64_t;
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using f32 = float;
using f64 = double;

using ggml_type = gguf::ggml_type;

static constexpr u32 GGML_MAX_DIMS = 4;
static constexpr u32 GGML_MAX_PARAMS = 2048;
static constexpr u32 GGML_MAX_NAME = 64;
static constexpr u32 GGML_MAX_SRC = 16;
static constexpr u32 GGML_MAX_OP_PARAMS = 64;

void* allocate(size_t size, size_t align = 16);
void deallocate(void* ptr, size_t align = 16);

//--- wyhash
//-----------------------------------------------------------
u32 wyhash32(size_t size, const void* key, u64 seed = 2685821657736338717ULL);
u64 wyhash64(size_t size, const void* key, u64 seed = 2685821657736338717ULL);

//--- prime numbers
//-----------------------------------------------------------
u32 next_prime(u32 x);

/**
 * @brief Find the first element in the range [first, last) which is more than equal val.
 */
template<class FwdIt, class T>
FwdIt lower_bound(FwdIt first, FwdIt last, const T& val)
{
    typename std::iterator_traits<FwdIt>::difference_type count = last - first;
    while(0 < count) {
        typename std::iterator_traits<FwdIt>::difference_type d = count / 2;
        FwdIt m = first + d;
        if(*m < val) {
            first = m + 1;
            count -= d + 1;
        } else {
            count = d;
        }
    }
    return first;
}

//--- Random
//-----------------------------------------------------------
class Random
{
public:
    Random();
    explicit Random(uint64_t seed);
    ~Random();

    void srand(uint64_t seed);

    /**
     * @brief Generate a 32bit unsigned value
     * @return
     */
    uint32_t rand();

    /**
     * @brief Generate a 32bit real number
     * @return [0 1)
     */
    float frand();

private:
    inline static constexpr uint64_t DEFAULT_SEED64 = 12345ULL;
    inline static constexpr uint64_t Increment = 1442695040888963407ULL;
    inline static constexpr uint64_t Multiplier = 6364136223846793005ULL;
    uint64_t state_;
};

//--- Array
//-----------------------------------------------------------
template<class T>
class Array
{
    static_assert(std::is_trivially_copyable<T>::value == true, "T should be trivially copyable.");

public:
    inline static constexpr uint64_t Expand = 64;
    Array();
    ~Array();
    Array(Array&& other);
    Array& operator=(Array&& other);
    uint64_t capacity() const;
    uint64_t size() const;
    const T& operator[](uint64_t index) const;
    T& operator[](uint64_t index);
    void clear();
    bool reserve(uint64_t capacity);
    bool resize(uint64_t capacity);
    bool push_back(const T& x);

private:
    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    bool expand(uint64_t capacity);
    uint64_t capacity_;
    uint64_t size_;
    T* items_;
};

template<class T>
Array<T>::Array()
    : capacity_(0)
    , size_(0)
    , items_(nullptr)
{
}

template<class T>
Array<T>::~Array()
{
    delete[] items_;
    items_ = nullptr;
}

template<class T>
Array<T>::Array(Array&& other)
    : capacity_(other.capacity_)
    , size_(other.size_)
    , items_(other.items_)
{
    other.capacity_ = 0;
    other.size_ = 0;
    other.items_ = nullptr;
}

template<class T>
Array<T>& Array<T>::operator=(Array&& other)
{
    if(this != &other) {
        delete[] items_;
        capacity_ = other.capacity_;
        size_ = other.size_;
        items_ = other.items_;
        other.capacity_ = 0;
        other.size_ = 0;
        other.items_ = nullptr;
    }
    return *this;
}

template<class T>
uint64_t Array<T>::capacity() const
{
    return capacity_;
}

template<class T>
uint64_t Array<T>::size() const
{
    return size_;
}

template<class T>
const T& Array<T>::operator[](uint64_t index) const
{
    assert(index < size_);
    return items_[index];
}

template<class T>
T& Array<T>::operator[](uint64_t index)
{
    assert(index < size_);
    return items_[index];
}

template<class T>
void Array<T>::clear()
{
    size_ = 0;
}

template<class T>
bool Array<T>::reserve(uint64_t capacity)
{
    capacity = (std::max)(size_, capacity);
    uint64_t new_capacity = Expand;
    while(new_capacity < capacity) {
        new_capacity += Expand;
    }
    return expand(new_capacity);
}

template<class T>
bool Array<T>::resize(uint64_t size)
{
    size = (std::max)(capacity_, size);
    uint64_t new_capacity = Expand;
    while(new_capacity < size) {
        new_capacity += Expand;
    }
    if(expand(new_capacity)) {
        assert(size <= new_capacity);
        size_ = size;
        return true;
    } else {
        return false;
    }
}

template<class T>
bool Array<T>::push_back(const T& x)
{
    if(capacity_ <= size_) {
        if(!expand(capacity_ + Expand)) {
            return false;
        }
        assert(size_ < capacity_);
    }
    items_[size_] = x;
    ++size_;
    return true;
}

template<class T>
bool Array<T>::expand(uint64_t capacity)
{
    if(capacity <= capacity_) {
        return true;
        ;
    }
    T* items = new T[capacity];
    if(nullptr != items_) {
        ::memcpy(items, items_, sizeof(T) * capacity);
        delete[] items_;
    }
    items_ = items;
    capacity_ = capacity;
    return true;
}

//--- PriorityQueue
//-----------------------------------------------------------
template<class T, class U=std::less<T>>
class PriorityQueue
{
    static_assert(std::is_trivially_copyable<T>::value == true, "T should be trivially copyable.");

public:
    inline static constexpr uint32_t Invalid = 0xFFFF'FFFFUL;
    inline static constexpr uint32_t Expand = 64;

    struct Iterator
    {
    public:
        Iterator(uint32_t capacity, uint32_t size, uint32_t pos, T* items);
        operator bool() const;
        void operator++();
        T& operator*();
        uint32_t position() const;
    private:
        uint32_t capacity_;
        uint32_t size_;
        uint32_t pos_;
        T* items_;
    };

    struct ReverseIterator
    {
    public:
        ReverseIterator(uint32_t capacity, uint32_t size, uint32_t pos, T* items);
        operator bool() const;
        void operator++();
        T& operator*();
        uint32_t position() const;
    private:
        uint32_t capacity_;
        uint32_t size_;
        uint32_t pos_;
        T* items_;
    };

    PriorityQueue();
    ~PriorityQueue();
    PriorityQueue(PriorityQueue&& other);
    PriorityQueue& operator=(PriorityQueue&& other);
    uint32_t capacity() const;
    uint32_t size() const;
    const T& operator[](uint32_t index) const;
    T& operator[](uint32_t index);
    void clear();
    bool reserve(uint32_t capacity);
    bool resize(uint32_t capacity);
    void push_back(const T& x);
    const T& front() const;
    T& front();
    void pop_front();
    Iterator begin();
    ReverseIterator rbegin();

private:
    PriorityQueue(const PriorityQueue&) = delete;
    PriorityQueue& operator=(const PriorityQueue&) = delete;

    bool expand(uint32_t capacity);
    uint32_t capacity_;
    uint32_t size_;
    uint32_t top_;
    uint32_t end_;
    U compare_;
    T* items_;
};

template<class T, class U>
PriorityQueue<T, U>::Iterator::Iterator(uint32_t capacity, uint32_t size, uint32_t pos, T* items)
    : capacity_(capacity)
    , size_(size)
    , pos_(pos)
    , items_(items)
{
}

template<class T, class U>
PriorityQueue<T, U>::Iterator::operator bool() const
{
    return 0<size_;
}

template<class T, class U>
void PriorityQueue<T, U>::Iterator::operator++()
{
    assert(0<size_);
    --size_;
    ++pos_;
    if(capacity_<=pos_){
        pos_ = 0;
    }
}

template<class T, class U>
T& PriorityQueue<T, U>::Iterator::operator*()
{
    assert(pos_<capacity_);
    return items_[pos_];
}

template<class T, class U>
uint32_t PriorityQueue<T, U>::Iterator::position() const
{
    return pos_;
}

template<class T, class U>
PriorityQueue<T, U>::ReverseIterator::ReverseIterator(uint32_t capacity, uint32_t size, uint32_t pos, T* items)
    : capacity_(capacity)
    , size_(size)
    , pos_(pos)
    , items_(items)
{
}

template<class T, class U>
PriorityQueue<T, U>::ReverseIterator::operator bool() const
{
    return 0<size_;
}

template<class T, class U>
void PriorityQueue<T, U>::ReverseIterator::operator++()
{
    assert(0<size_);
    --size_;
    if(pos_<=0){
        pos_ = capacity_-1;
    }else{
        --pos_;
    }
}

template<class T, class U>
T& PriorityQueue<T, U>::ReverseIterator::operator*()
{
    assert(pos_<capacity_);
    return items_[pos_];
}

template<class T, class U>
uint32_t PriorityQueue<T, U>::ReverseIterator::position() const
{
    return pos_;
}

template<class T, class U>
PriorityQueue<T,U>::PriorityQueue()
    : capacity_(0)
    , size_(0)
    , top_(0)
    , end_(0)
    , items_(nullptr)
{
}

template<class T, class U>
PriorityQueue<T,U>::~PriorityQueue()
{
    capacity_ = 0;
    size_ = 0;
    top_ = 0;
    end_ = 0;
    delete[] items_;
    items_ = nullptr;
}

template<class T, class U>
PriorityQueue<T,U>::PriorityQueue(PriorityQueue&& other)
    : capacity_(other.capacity_)
    , size_(other.size_)
    , top_(other.top_)
    , end_(other.end_)
    , items_(other.items_)
{
    other.capacity_ = 0;
    other.size_ = 0;
    other.top_ = 0;
    other.end_ = 0;
    other.items_ = nullptr;
}

template<class T, class U>
PriorityQueue<T,U>& PriorityQueue<T,U>::operator=(PriorityQueue&& other)
{
    if(this != &other) {
        delete[] items_;
        capacity_ = other.capacity_;
        size_ = other.size_;
        top_ = other.top_;
        end_ = other.end_;
        items_ = other.items_;
        other.capacity_ = 0;
        other.size_ = 0;
        other.top_ = 0;
        other.end_ = 0;
        other.items_ = nullptr;
    }
    return *this;
}

template<class T, class U>
uint32_t PriorityQueue<T,U>::capacity() const
{
    return capacity_;
}

template<class T, class U>
uint32_t PriorityQueue<T,U>::size() const
{
    return size_;
}

template<class T, class U>const T& PriorityQueue<T,U>::operator[](uint32_t index) const
{
    assert(index < size_);
    return items_[index];
}

template<class T, class U>
T& PriorityQueue<T,U>::operator[](uint32_t index)
{
    assert(index < size_);
    return items_[index];
}

template<class T, class U>
void PriorityQueue<T,U>::clear()
{
    size_ = 0;
}

template<class T, class U>
bool PriorityQueue<T,U>::reserve(uint32_t capacity)
{
    capacity = (std::max)(size_, capacity);
    uint64_t new_capacity = Expand;
    while(new_capacity < capacity) {
        new_capacity += Expand;
    }
    return expand(new_capacity);
}

template<class T, class U>
bool PriorityQueue<T,U>::resize(uint32_t size)
{
    size = (std::max)(capacity_, size);
    uint64_t new_capacity = Expand;
    while(new_capacity < size) {
        new_capacity += Expand;
    }
    if(expand(new_capacity)) {
        assert(size <= new_capacity);
        size_ = size;
        return true;
    } else {
        return false;
    }
}

template<class T, class U>
void PriorityQueue<T,U>::push_back(const T& x)
{
    if(capacity_ <= size_) {
        if(!expand(capacity_ + Expand)) {
            return;
        }
        assert(size_ < capacity_);
    }
    assert(end_<capacity_);
    items_[end_] = x;
    end_ = capacity_<=(end_+1)? 0 : (end_+1);
    ++size_;
    for(ReverseIterator itr = rbegin(); itr;){
        uint32_t p0 = itr.position();
        ++itr;
        if(!itr){
            break;
        }
        uint32_t p1 = itr.position();
        if(compare_(items_[p0], items_[p1])){
            break;
        }
        (std::swap)(items_[p0], items_[p1]);
    }
}

template<class T, class U>
const T& PriorityQueue<T, U>::front() const
{
    assert(0<size_);
    return items_[top_];
}

template<class T, class U>
T& PriorityQueue<T, U>::front()
{
    assert(0<size_);
    return items_[top_];
}

template<class T, class U>
void PriorityQueue<T, U>::pop_front()
{
    assert(0<size_);
    top_ = capacity_<=(top_+1)? 0 : (top_+1);
    --size_;
}

template<class T, class U>
PriorityQueue<T, U>::Iterator PriorityQueue<T, U>::begin()
{
    uint32_t pos = (0<end_)? end_-1 : capacity_-1;
    return Iterator(capacity_, size_, pos, items_);
}

template<class T, class U>
PriorityQueue<T, U>::ReverseIterator PriorityQueue<T, U>::rbegin()
{
    uint32_t pos = (0<end_)? end_-1 : capacity_-1;
    return ReverseIterator(capacity_, size_, pos, items_);
}

template<class T, class U>
bool PriorityQueue<T,U>::expand(uint32_t capacity)
{
    if(capacity <= capacity_) {
        return true;
    }
    T* items = new T[capacity];

    if(0 < size_) {
        if(end_<=top_) {
            assert(top_ == end_);
            assert(size_ == capacity_);
            assert(top_<capacity_);
            ::memcpy(items, items_+top_, (capacity_-top_)*sizeof(T));
            if(0 < end_) {
                ::memcpy(items + top_, items_, end_*sizeof(T));
            }
        } else {
            ::memcpy(items, items_ + top_, size_*sizeof(T));
        }
        top_ = 0;
        end_ = size_;
    }
    delete[] items_;
    items_ = items;
    capacity_ = capacity;
    return true;
}

//--- Hasher
//-----------------------------------------------------------
template<class T>
struct Hasher
{
    u32 operator()(const T& x) const;
};

//--- HashMap
//-----------------------------------------------------------
template<class T, class U>
class HashMap
{
public:
    inline static constexpr u32 Invalid = 0xFFFF'FFFFUL;

    HashMap();
    explicit HashMap(u32 capacity);
    ~HashMap();
    HashMap(HashMap<T, U>&& other);
    HashMap& operator=(HashMap<T, U>&& other);
    void reserve(u32 capacity);
    u32 size() const;
    void clear();
    void add(const T& key, const U& value);
    void remove(const T& key);
    void remove(u32 pos);
    u32 find(const T& key) const;
    bool tryGet(const T& key, const U*& value) const;
    bool tryGet(const T& key, U*& value);

    void swap(HashMap& other);

    u32 begin() const;
    u32 end() const;

    u32 next(u32 pos) const;

    U& getValue(u32 pos);
    const U& getValue(u32 pos) const;

    T& getKey(u32 pos);
    const T& getKey(u32 pos) const;

private:
    HashMap(const HashMap&) = delete;
    HashMap& operator=(const HashMap&) = delete;

    inline static u64 align(u64 x)
    {
        return (x + 7ULL) & (~7ULL);
    }

    void expand();
    void create(u32 capacity);
    u32 find(const T& key, u32 hash) const;
    void remove(u32 pos, u32 hash);

    struct Entry
    {
        inline static constexpr u32 HashMask = 0x7FFFFFFFU;
        inline static constexpr u32 OccupyFlag = 0x80000000U;

        bool isOccupy() const
        {
            return 0 != (hash_ & OccupyFlag);
        }
        void setOccupy()
        {
            hash_ |= OccupyFlag;
        }
        void setEmpty()
        {
            hash_ &= HashMask;
        }

        u32 index_;
        u32 next_;
        u32 hash_;
    };

    u32 capacity_;
    u32 size_;
    u32 empty_;
    Hasher<T> hasher_;
    Entry* entries_;
    T* keys_;
    U* values_;
};

template<class T, class U>
HashMap<T, U>::HashMap()
    : capacity_(0)
    , size_(0)
    , empty_(Invalid)
    , entries_(nullptr)
    , keys_(nullptr)
    , values_(nullptr)
{
}

template<class T, class U>
HashMap<T, U>::HashMap(u32 capacity)
    : capacity_(0)
    , size_(0)
    , empty_(Invalid)
    , entries_(nullptr)
    , keys_(nullptr)
    , values_(nullptr)
{
    create(capacity);
}

template<class T, class U>
HashMap<T, U>::~HashMap()
{
    clear();
    deallocate(entries_);
    capacity_ = 0;
    entries_ = nullptr;
    keys_ = nullptr;
    values_ = nullptr;
}

template<class T, class U>
HashMap<T, U>::HashMap(HashMap<T, U>&& other)
    : capacity_(other.capacity_)
    , size_(other.size_)
    , empty_(other.empty_)
    , hasher_(other.hasher_)
    , entries_(other.entries_)
    , keys_(other.keys_)
    , values_(other.values_)
{
    other.capacity_ = 0;
    other.size_ = 0;
    other.empty_ = Invalid;
    other.entries_ = nullptr;
    other.keys_ = nullptr;
    other.values_ = nullptr;
}

template<class T, class U>
HashMap<T, U>& HashMap<T, U>::operator=(HashMap<T, U>&& other)
{
    if(this != &other) {
        clear();
        deallocate(entries_);

        capacity_ = other.capacity_;
        size_ = other.size_;
        empty_ = other.empty_;
        hasher_ = other.hasher_;
        entries_ = other.entries_;
        keys_ = other.keys_;
        values_ = other.values_;

        other.capacity_ = 0;
        other.size_ = 0;
        other.empty_ = Invalid;
        other.entries_ = nullptr;
        other.keys_ = nullptr;
        other.values_ = nullptr;
    }
    return *this;
}

template<class T, class U>
void HashMap<T, U>::reserve(u32 capacity)
{
    clear();
    deallocate(entries_);
    capacity_ = 0;
    entries_ = nullptr;
    keys_ = nullptr;
    values_ = nullptr;
    create(capacity);
}

template<class T, class U>
u32 HashMap<T, U>::size() const
{
    return size_;
}

template<class T, class U>
void HashMap<T, U>::clear()
{
    for(u32 i = 0; i < capacity_; ++i) {
        if(entries_[i].isOccupy()) {
            keys_[i].~T();
            values_[i].~U();
            entries_[i].hash_ = 0;
            entries_[i].index_ = Invalid;
            entries_[i].next_ = empty_;
            empty_ = i;
        }
    }
    size_ = 0;
}

template<class T, class U>
void HashMap<T, U>::add(const T& key, const U& value)
{
    u32 hash = hasher_(key);
    if(0 < capacity_ && find(key, hash) != end()) {
        return;
    }

    if(empty_ == Invalid) {
        expand();
        assert(Invalid != empty_);
    }
    u32 pos = empty_;
    empty_ = entries_[empty_].next_;

    u32 entryPos = hash % capacity_;
    entries_[pos].next_ = entries_[entryPos].index_;
    entries_[entryPos].index_ = pos;
    entries_[pos].hash_ = hash | Entry::OccupyFlag;
    new(&keys_[pos]) T(key);
    new(&values_[pos]) U(value);
    ++size_;
}

template<class T, class U>
void HashMap<T, U>::remove(const T& key)
{
    if(capacity_ <= 0) {
        return;
    }
    u32 hash = hasher_(key);
    u32 pos = find(key, hash);
    if(end() == pos || !entries_[pos].isOccupy()) {
        return;
    }
    remove(pos, hash);
}

template<class T, class U>
void HashMap<T, U>::remove(u32 pos)
{
    if(capacity_ <= 0) {
        return;
    }
    if(end() == pos || !entries_[pos].isOccupy()) {
        return;
    }
    u32 hash = hasher_(getKey(pos));
    remove(pos, hash);
}

template<class T, class U>
u32 HashMap<T, U>::find(const T& key) const
{
    return (0 < capacity_) ? find(key, hasher_(key)) : end();
}
template<class T, class U>
bool HashMap<T, U>::tryGet(const T& key, const U*& value) const
{
    u32 pos = find(key);
    if(pos != end()) {
        value = &getValue(pos);
        return true;
    }
    return false;
}

template<class T, class U>
bool HashMap<T, U>::tryGet(const T& key, U*& value)
{
    u32 pos = find(key);
    if(pos != end()) {
        value = &getValue(pos);
        return true;
    }
    return false;
}

template<class T, class U>
void HashMap<T, U>::swap(HashMap& other)
{
    std::swap(capacity_, other.capacity_);
    std::swap(size_, other.size_);
    std::swap(empty_, other.empty_);
    std::swap(entries_, other.entries_);
    std::swap(keys_, other.keys_);
    std::swap(values_, other.values_);
}

template<class T, class U>
u32 HashMap<T, U>::begin() const
{
    for(u32 i = 0; i < capacity_; ++i) {
        if(entries_[i].isOccupy()) {
            return i;
        }
    }
    return end();
}

template<class T, class U>
u32 HashMap<T, U>::end() const
{
    return Invalid;
}

template<class T, class U>
u32 HashMap<T, U>::next(u32 pos) const
{
    for(u32 i = pos + 1; i < capacity_; ++i) {
        if(entries_[i].isOccupy()) {
            return i;
        }
    }
    return end();
}

template<class T, class U>
U& HashMap<T, U>::getValue(u32 pos)
{
    assert(pos < capacity_);
    return values_[pos];
}

template<class T, class U>
const U& HashMap<T, U>::getValue(u32 pos) const
{
    assert(pos < capacity_);
    return values_[pos];
}

template<class T, class U>
T& HashMap<T, U>::getKey(u32 pos)
{
    assert(pos < capacity_);
    return keys_[pos];
}

template<class T, class U>
const T& HashMap<T, U>::getKey(u32 pos) const
{
    assert(pos < capacity_);
    return keys_[pos];
}

template<class T, class U>
void HashMap<T, U>::expand()
{
    HashMap<T, U> tmp(capacity_ + 1);

    for(u32 i = 0; i < capacity_; ++i) {
        if(entries_[i].isOccupy()) {
            tmp.add(keys_[i], values_[i]);
        }
    }
    tmp.swap(*this);
}

template<class T, class U>
void HashMap<T, U>::create(u32 capacity)
{
    capacity_ = next_prime(capacity);
    u64 entry_size = align(sizeof(Entry) * capacity_);
    u64 key_size = align(sizeof(T) * capacity_);
    u64 value_size = align(sizeof(U) * capacity_);
    u8* buffer = static_cast<u8*>(allocate(entry_size + key_size + value_size));
    Entry* entries = reinterpret_cast<Entry*>(buffer);
    T* keys = reinterpret_cast<T*>(buffer + entry_size);
    U* values = reinterpret_cast<U*>(buffer + entry_size + key_size);
    for(u32 i = 0; i < capacity_; ++i) {
        entries[i].index_ = Invalid;
        entries[i].next_ = i + 1;
        entries[i].hash_ = 0;
    }
    entries[capacity_ - 1].next_ = Invalid;
    empty_ = 0;
    deallocate(entries_);
    entries_ = entries;
    keys_ = keys;
    values_ = values;
}

template<class T, class U>
u32 HashMap<T, U>::find(const T& key, u32 hash) const
{
    assert(0 < capacity_);
    u32 pos = hash % capacity_;
    hash |= Entry::OccupyFlag;
    for(u32 i = entries_[pos].index_; Invalid != i; i = entries_[i].next_) {
        if(hash == entries_[i].hash_ && key == keys_[i]) {
            return i;
        }
    }
    return end();
}

template<class T, class U>
void HashMap<T, U>::remove(u32 pos, u32 hash)
{
    assert(0 < size_);
    u32 entryPos = hash % capacity_;

    keys_[pos].~T();
    values_[pos].~U();

    if(pos == entries_[entryPos].index_) {
        entries_[entryPos].index_ = entries_[pos].next_;
    } else {
        u32 p;
        for(p = entries_[entryPos].index_; entries_[p].next_ != pos; p = entries_[p].next_) {
#if _DEBUG
            if(entries_[p].next_ == Invalid) {
                assert(false);
            }
#endif
        }
        entries_[p].next_ = entries_[pos].next_;
    }
    entries_[pos].next_ = empty_;
    empty_ = pos;
    entries_[pos].hash_ = 0;
    --size_;
}

namespace util
{
    void copy1(u32 bits, u64 size, void* dst, const void* src);
    void copy2(u32 bits, u64 size, void* dst, const void* src);
    void copy3(u32 bits, u64 size, void* dst, const void* src);
    void copy4(u32 bits, u64 size, void* dst, const void* src);
    void copy5(u32 bits, u64 size, void* dst, const void* src);
    void copy6(u32 bits, u64 size, void* dst, const void* src);
    void copy8(u32 bits, u64 size, void* dst, const void* src);
    void copy16(u32 bits, u64 size, void* dst, const void* src);
    void copy32(u32 bits, u64 size, void* dst, const void* src);
    void copy64(u32 bits, u64 size, void* dst, const void* src);

    void copy1_f(u64 size, void* dst, const void* src);
    void copy2_f(u64 size, void* dst, const void* src);
    void copy3_f(u64 size, void* dst, const void* src);
    void copy4_f(u64 size, void* dst, const void* src);
    void copy5_f(u64 size, void* dst, const void* src);
    void copy6_f(u64 size, void* dst, const void* src);
    void copy8_f(u64 size, void* dst, const void* src);
    void copyi8_f(u64 size, void* dst, const void* src);
    void copyi16_f(u64 size, void* dst, const void* src);
    void copyi32_f(u64 size, void* dst, const void* src);
    void copyi64_f(u64 size, void* dst, const void* src);
    void copyf16_f(u64 size, void* dst, const void* src);
    void copyf32_f(u64 size, void* dst, const void* src);
    void copyf64_f(u64 size, void* dst, const void* src);
} // namespace util

//--- Timer
//-----------------------------------------------------------
class Timer
{
public:
    Timer(s64& duration);
    ~Timer();

private:
    s64& duration_;
    std::chrono::high_resolution_clock::time_point start_;
};

struct Config;
struct Context;

//--- Tensor
//-----------------------------------------------------------
class Tensor
{
public:
    Tensor();
    Tensor(ggml_type type, std::initializer_list<u64> dimensions);
    Tensor(ggml_type type, std::initializer_list<u64> dimensions, const void* data);
    Tensor(ggml_type type, const Tensor& shape);
    Tensor(ggml_type type, const Tensor& shape, const void* data);
    Tensor(Tensor&& other);
    ~Tensor();
    Tensor& operator=(Tensor&& other);
    ggml_type type() const;
    u32 num_dims() const;
    u64 total_size() const;
    u64 total_bytes() const;
    u64 size(u32 index) const;

    template<typename T>
    const T* data() const
    {
        return reinterpret_cast<const T*>(data_.get());
    }

    template<typename T>
    T* data()
    {
        return reinterpret_cast<T*>(data_.get());
    }

    void resize(std::initializer_list<u64> dimensions) noexcept;

private:
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    friend bool is_same_shape(const Tensor&, const Tensor&);

    struct CustomDeleter
    {
        constexpr CustomDeleter(bool dummy) noexcept
            : dummy_(dummy)
        {
        }

        void operator()(u8* ptr) const
        {
            if(dummy_) {
                return;
            }
            delete[] ptr;
        }
        bool dummy_;
    };

    ggml_type type_;
    u16 num_dims_;
    u16 bit_packed_;
    u64 dims_[GGML_MAX_DIMS];
    std::unique_ptr<u8[], CustomDeleter> data_;
};

namespace op
{
    Tensor convertF32(const Tensor& input);
    f32 kahan_sum(u64 size, const f32* src);
    f32 kahan_sum_squared(u64 size, const f32* src, f32 mean);
    void normalize_vec(u64 size, f32* dst, const f32* src, const f32* weight, const f32* bias);
    Tensor normalize(const Tensor& input, const Tensor& weight, const Tensor& bias);
    Tensor embed_tokens(const Tensor& emb_weight, const Tensor& tokens);
    Tensor embed_projection(const Tensor& input, const Tensor& emb_weight);
    Tensor gelu(const Tensor& input);
    void vec_add(u64 size, f32* dst, const f32* src0, const f32* src1);
    Tensor add(const Tensor& x0, const Tensor& x1);
    Tensor affine_proj_2d(const Tensor& input, const Tensor& weight);
    Tensor affine_proj_2d(const Tensor& input, const Tensor& weight, const Tensor& bias);
    void matmul(f32* dst,const f32* x, const f32* w, u64 n, u64 d);
    void rmsnorm(u64 size, f32* dst, const f32* x, const f32* w, f32 epsilon);
} // namespace op

bool is_same_shape(const Tensor& x0, const Tensor& x1);

//--- Embedding
//-----------------------------------------------------------
class Embedding
{
public:
    Embedding();
    Embedding(
        ggml_type type,
        u32 n_vocab,
        u32 d_embed,
        void* weight);
    ~Embedding();
    Embedding(Embedding&& other);
    Embedding& operator=(Embedding&& other);

    Tensor forward(const Tensor& input);
    Tensor forward_proj(const Tensor& input);

    inline s64 time() const
    {
        return duration_;
    }

private:
    s64 duration_;
    Tensor weight_;
};

//--- PositionalEmbedding
//-----------------------------------------------------------
class PositionalEmbedding
{
public:
    PositionalEmbedding();
    PositionalEmbedding(
        ggml_type type,
        u64 max_context,
        u64 d_embed,
        void* weight);
    ~PositionalEmbedding();
    PositionalEmbedding(PositionalEmbedding&& other);
    PositionalEmbedding& operator=(PositionalEmbedding&& other);

    Tensor forward(u64 num_context);

    inline s64 time() const
    {
        return duration_;
    }

private:
    s64 duration_;
    Tensor weight_;
};

//--- Residual
//-----------------------------------------------------------
class Residual
{
public:
    Residual();
    ~Residual();

    void forward(Tensor& dst, const Tensor& src0, const Tensor& src1);
    inline s64 time() const
    {
        return duration_;
    }
private:
    s64 duration_;
};

//--- RMSNorm
//-----------------------------------------------------------
class RMSNorm
{
public:
    RMSNorm();
    RMSNorm(Tensor&& weight, f32 epsilon = 1.0e-5);
    ~RMSNorm();
    RMSNorm(RMSNorm&& other);
    RMSNorm& operator=(RMSNorm&& other);

    void forward(Tensor& dst, const Tensor& src);
    inline s64 time() const
    {
        return duration_;
    }
private:
    RMSNorm(const RMSNorm& other) = delete;
    RMSNorm& operator=(const RMSNorm& other) = delete;

    s64 duration_;
    f32 epsilon_;
    Tensor weight_;
};

//--- SelfAttention
//-----------------------------------------------------------
class SelfAttention
{
public:
    SelfAttention();
    SelfAttention(
        Tensor&& query,
        Tensor&& key,
        Tensor&& value,
        Tensor&& qkv);
    ~SelfAttention();
    SelfAttention(SelfAttention&& other);
    SelfAttention& operator=(SelfAttention&& other);

    void forward(
        const Config& config,
        u64 position,
        u64 layer_offset,
        Tensor& output,
        Tensor& input,
        Tensor& query,
        Tensor& key_cache,
        Tensor& value_cache,
        Tensor& attention);

    inline s64 time() const
    {
        return duration_;
    }

private:
    SelfAttention(const SelfAttention&) = delete;
    SelfAttention& operator=(const SelfAttention&) = delete;
    s64 duration_;
    Tensor query_;
    Tensor key_;
    Tensor value_;
    Tensor qkv_proj_;
};

//--- FeedForwardSwiGLU
//-----------------------------------------------------------
class FeedForwardSwiGLU
{
public:
    FeedForwardSwiGLU();
    FeedForwardSwiGLU(
        Tensor&& ffn_down,
        Tensor&& ffn_gate,
        Tensor&& ffn_up,
        Tensor&& ffn_norm);
    ~FeedForwardSwiGLU();
    FeedForwardSwiGLU(FeedForwardSwiGLU&& other);
    FeedForwardSwiGLU& operator=(FeedForwardSwiGLU&& other);

    void forward(
        const Config& config,
        Tensor& output,
        const Tensor& input,
        Tensor& buffer0,
        Tensor& buffer1);

    inline s64 time() const
    {
        return duration_;
    }

private:
    FeedForwardSwiGLU(const FeedForwardSwiGLU&) = delete;
    FeedForwardSwiGLU& operator=(const FeedForwardSwiGLU&) = delete;
    s64 duration_;
    Tensor ffn_down_; // w2 of safetensor
    Tensor ffn_gate_; // w1 of safetensor
    Tensor ffn_up_; // w3 of safetensor
    Tensor ffn_norm_;
};

//--- TransformerBlock
//-----------------------------------------------------------
class TransformerBlock
{
public:
    TransformerBlock();
    ~TransformerBlock();
    TransformerBlock(TransformerBlock&& other);
    TransformerBlock& operator=(TransformerBlock&& other);

    void forward(
        const Config& config,
        u64 layer,
        u64 position,
        Tensor& output,
        Tensor& input,
        Tensor& query,
    Tensor& key_cache,
    Tensor& value_cache,
        Tensor& attention,
        Tensor& buffer0,
        Tensor& buffer1,
        Tensor& hbuffer0,
        Tensor& hbuffer1);

    inline s64 time() const
    {
        return duration_;
    }

private:
    TransformerBlock(const TransformerBlock&) = delete;
    TransformerBlock& operator=(const TransformerBlock&) = delete;
    s64 duration_;
    RMSNorm attn_rmsnorm_;
    SelfAttention attn_;
    Residual attn_residual_;
    RMSNorm ff_rmsnorm_;
    FeedForwardSwiGLU ff_;
    Residual ff_residual_;
};

//--- String
//-----------------------------------------------------------
struct String
{
    u64 len_;
    const char8_t* str_;
    bool operator==(const String& x) const
    {
        return len_==x.len_ && 0 == ::strncmp((const char*)str_, (const char*)x.str_, len_);
    }
};

template<>
struct Hasher<String>
{
    u32 operator()(const String& x) const
    {
        return wyhash32(x.len_, x.str_);
    }
};

template<>
struct Hasher<u32>
{
    u32 operator()(const u32& x) const
    {
        return wyhash32(sizeof(u32), &x);
    }
};

//--- Vocabulary
//-----------------------------------------------------------
class Vocabulary
{
public:
    struct Token
    {
        String text_;
        f32 score_;
        s32 type_;
    };

    inline static constexpr u32 Invalid = 0xFFFF'FFFFUL;

    Vocabulary();
    Vocabulary(const gguf::GGUF& model_data);
    ~Vocabulary();
    Vocabulary(Vocabulary&& other);
    Vocabulary& operator=(Vocabulary&& other);

    const gguf::GGUFString& getModel() const;
    const gguf::GGUFArray& getTokens() const;
    const gguf::GGUFArray& getScores() const;
    const gguf::GGUFArray& getTokenTypes() const;
    const gguf::GGUFArray& getMerges() const;
    const gguf::GGUFArray& getAddedTokens() const;
    s32 getBOS() const;
    s32 getEOS() const;
    s32 getUnknown() const;
    s32 getSeparator() const;
    s32 getPadding() const;
    s32 getCls() const;
    s32 getMask() const;
    s32 getMaxTokenLength() const;
    u64 idToTokenSize() const;
    const Token& idToToken(s32 x) const;
    bool tokenToId(s32& id, const String& token) const;

    bool encode(s32& token, const char8_t* str) const;
    bool encode(s32& token, u64 length, const char8_t* str) const;
    bool decode(char8_t str[512], s32 token) const;
    bool decode(u64 length, char8_t str[], s32 token) const;
private:
    Vocabulary(const Vocabulary&) = delete;
    Vocabulary& operator=(const Vocabulary&) = delete;
    gguf::GGUFString model_;
    gguf::GGUFArray tokens_;
    gguf::GGUFArray scores_;
    gguf::GGUFArray token_types_;
    gguf::GGUFArray merges_;
    gguf::GGUFArray added_tokens_;

    s32 bos_token_id_;
    s32 eos_token_id_;
    s32 unknown_token_id_;
    s32 separator_token_id_;
    s32 padding_token_id_;
    s32 cls_token_id_;
    s32 mask_token_id_;

    s32 add_bos_;
    s32 add_eos_;
    
    s32 linefeed_id_;
    s32 prefix_id_;
    s32 suffix_id_;
    s32 middle_id_;
    s32 eot_id_;

    bool add_space_prefix_;

    s32 max_token_length_;
    HashMap<String, s32> tokenToId_;
    Array<Token> idToToken_;
    Array<s32> cache_special_tokens_;
    Array<String> cache_token_to_piece_;
};

//--- Tokenizer
//-----------------------------------------------------------
class Tokenizer
{
public:
    Tokenizer();
    Tokenizer(const gguf::GGUF& model_data);
    ~Tokenizer();
    Tokenizer(Tokenizer&& other) noexcept;
    Tokenizer& operator=(Tokenizer&& other);
    Array<s32> tokenize(const std::u8string& text);

private:
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    struct Symbol
    {
        using index = s32;
        index prev_;
        index next_;
        u64 len_;
        const char8_t* text_;
    };

    struct Bigram
    {
        struct Comparator
        {
            bool operator()(Bigram& x0, Bigram& x1) const
            {
                return (x0.score_ < x1.score_)
                    || (x0.score_ == x1.score_ && x0.left_ > x1.left_);
            }
        };
        Symbol::index left_;
        Symbol::index right_;
        float score_;
        u64 size_;
    };

    struct Pair
    {
        Symbol::index left_;
        Symbol::index right_;
    };

    static u64 length(char c);
    static unicode_byte_to_utf8_map();
    static s32 byte_to_token(const Vocabulary& vocab, char8_t c);
    void try_add_bigram(Symbol::index left, Symbol::index right);
    void resegment(Array<s32>& output, const Symbol& symbol) const;

    static const char8_t* Pattern;
    char8_t* buffer_;
    Vocabulary vocab_;
    Array<Symbol> symbols_;
    PriorityQueue<Bigram> work_queue_;
    HashMap<String, Pair> rev_merge_;
};

//--- Sampler
//-----------------------------------------------------------
class Sampler
{
public:
    Sampler();
    ~Sampler();
    Sampler(Sampler&& other);
    Sampler& operator=(Sampler&& other);

    u32 sample(f32* logits);
private:
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    struct ProbIndex
    {
        f32 prob_;
        u32 index_;
    };

    static u32 sample_argmax(u32 size, const f32* probabilities);
    static u32 sample_mult(u32 size, f32 coin, const f32* probabilities);
    static u32 sample_topp(u32 size, f32 topp, f32 coin, ProbIndex* probindex, const f32* probabilities);
    u32 vocab_size_;
    f32 temperature_;
    f32 topp_;
    Random random_;
    ProbIndex* probindex_;
};

struct Config
{
    u64 dimension_;
    u64 hidden_dim_;
    u64 num_layers_;
    u64 num_heads_;
    u64 num_kv_heads_;
    u64 vocab_size_;
    u64 sequence_length_;
};

struct Context
{
    Tensor x_; // activation at current time stamp
    Tensor xb_; // activation, but inside a resdual branch
    Tensor xb2_; // additional buffer
    Tensor hb_; // buffer for hidden dimension in the ffn
    Tensor hb2_; // buffer for hidden dimension in the ffn
    Tensor query_; // query
    Tensor attn_; // buffer for scores/attention values
    Tensor logits_; // output logits
    Tensor key_cache_;
    Tensor value_cache_;
};

//--- Llama2
//-----------------------------------------------------------
class Llama2
{
public:
    /*
    llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}
    */
    Llama2();
    explicit Llama2(const Config& config);
    Llama2(Llama2&& other);
    virtual ~Llama2();
    Llama2& operator=(Llama2&& other);

private:
    Llama2(const Llama2&) = delete;
    Llama2& operator=(const Llama2&) = delete;
    void forward(u32 token, u32 position);

    Config config_;
    Sampler sampler_;
    Context context_;
    TransformerBlock* blocks_;
    RMSNorm output_rmsnorm_;
    Tensor output_weight_;
};
} // namespace cppgpt
#endif // INC_CPPGPT_H_

#ifndef INC_CPPGPT_H_
#define INC_CPPGPT_H_
#include "gguf.h"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <re2/re2.h>
#include <type_traits>
#include <utility>
#include <istream>

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

//--- Array
//-----------------------------------------------------------
template<class T>
class Array
{
    static_assert(std::is_trivially_copyable<T>::value == true, "T should be trivially copyable.");
public:
    inline static constexpr uint64_t Expand = 16;
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
    if(this != &other){
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
    while(new_capacity<capacity){
        new_capacity += Expand;
    }
    return expand(new_capacity);
}

template<class T>
bool Array<T>::resize(uint64_t size)
{
    size = (std::max)(capacity_, size);
    uint64_t new_capacity = Expand;
    while(new_capacity<size){
        new_capacity += Expand;
    }
    if(expand(new_capacity)){
        assert(size<new_capacity);
        size_ = size;
        return true;
    }else{
        return false;
    }
}

template<class T>
bool Array<T>::push_back(const T& x)
{
    if(capacity_<=size_){
        if(!expand(capacity_+Expand)){
            return false;
        }
        assert(size_<capacity_);
    }
    items_[size_] = x;
    ++size_;
    return true;
}

template<class T>
bool Array<T>::expand(uint64_t capacity)
{
    if(capacity<=capacity_){
        return true;;
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
    u32 size() const;
    void clear();
    void add(const T& key, const U& value);
    void remove(const T& key);
    void remove(u32 pos);
    u32 find(const T& key) const;
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
    :capacity_(other.capacity_)
    ,size_(other.size_)
    ,empty_(other.empty_)
    ,hasher_(other.hasher_)
    ,entries_(other.entries_)
    ,keys_(other.keys_)
    ,values_(other.values_)
{
    other.capacity_ = 0;
    other.size_ = 0;
    other.empty_ = Invalid;
    other.entries_ = nullptr;
    other.keys_ = nullptr;
    other.values_ = nullptr;
}

template<class T, class U>
    HashMap<T,U>& HashMap<T, U>::operator=(HashMap<T, U>&& other)
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

//--- Memory
//-----------------------------------------------------------
class Memory
{
public:
    Memory();
    explicit Memory(u64 size);
    Memory(Memory&& other);
    ~Memory();
    Memory& operator=(Memory&& other);

    u64 size() const
    {
        return size_;
    }

    template<class T>
    operator const T*() const
    {
        return reinterpret_cast<const T*>(ptr_);
    }

    template<class T>
    operator T*()
    {
        return reinterpret_cast<T*>(ptr_);
    }

private:
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    u64 size_;
    void* ptr_;
};

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
    Tensor affine_proj_2d(const Tensor& input, const Tensor& weight, const Tensor& bias);
} // namespace op

bool is_same_shape(const Tensor& x0, const Tensor& x1);

//--- LayerNorm
//-----------------------------------------------------------
class LayerNorm
{
public:
    LayerNorm();
    LayerNorm(
        ggml_type type,
        u32 d_in,
        u32 d_out,
        void* weight,
        void* bias);
    ~LayerNorm();
    LayerNorm(LayerNorm&& other);
    LayerNorm& operator=(LayerNorm&& other);

    Tensor forward(const Tensor& input);

    inline s64 time() const
    {
        return duration_;
    }

private:
    s64 duration_;
    Tensor weight_;
    Tensor bias_;
};

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

//--- GELU
//-----------------------------------------------------------
class GELU
{
public:
    GELU();
    ~GELU();
    GELU(GELU&& other);
    GELU& operator=(GELU&& other);

    Tensor forward(const Tensor& input);

    inline s64 time() const
    {
        return duration_;
    }

private:
    s64 duration_;
};

//--- Residual
//-----------------------------------------------------------
class Residual
{
public:
    Residual();
    ~Residual();
    Residual(Residual&& other);
    Residual& operator=(Residual&& other);

    Tensor forward(const Tensor& input0, const Tensor& input1);

    inline s64 time() const
    {
        return duration_;
    }

private:
    s64 duration_;
};

//--- Linear
//-----------------------------------------------------------
class Linear
{
public:
    Linear();
    Linear(ggml_type type, u64 d_in, u64 d_out, void* weight, void* bias);
    ~Linear();
    Linear(Linear&& other);
    Linear& operator=(Linear&& other);

    Tensor forward(const Tensor& input);

    inline s64 time() const
    {
        return duration_;
    }

private:
    s64 duration_;
    Tensor weight_;
    Tensor bias_;
};

//--- MultiHeadSelfAttn
//-----------------------------------------------------------
class MultiHeadSelfAttn
{
public:
    MultiHeadSelfAttn();
    MultiHeadSelfAttn(
        ggml_type type,
        u64 n_heads,
        u64 n_embed,
        void* query_w,
        void* query_b,
        void* key_w,
        void* key_b,
        void* value_w,
        void* value_b,
        void* qkv_proj_w,
        void* qkv_proj_b);
    ~MultiHeadSelfAttn();
    MultiHeadSelfAttn(MultiHeadSelfAttn&& other);
    MultiHeadSelfAttn& operator=(MultiHeadSelfAttn&& other);

    Tensor forward(const Tensor& input);

    inline s64 time() const
    {
        return duration_;
    }

private:
    Tensor masked_qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v);

    s64 duration_;
    u64 n_heads_;
    Linear query_;
    Linear key_;
    Linear value_;
    Linear qkv_proj_;
};

class ResidualAttnBlock
{
public:
    ResidualAttnBlock();
    ~ResidualAttnBlock();
    ResidualAttnBlock(ResidualAttnBlock&& other);
    ResidualAttnBlock& operator=(ResidualAttnBlock&& other);

    Tensor forward(const Tensor& input);
    inline s64 time() const
    {
        return duration_;
    }

private:
    s64 duration_;
    LayerNorm attn_ln_;
    MultiHeadSelfAttn attn_;
    Residual inp_res_;
    LayerNorm mlp_ln_;
    Linear mlp_fc_;
    GELU gelu_;
    Linear mlp_proj_;
    Residual attn_res_;
};

//--- RMSNorm
//-----------------------------------------------------------
class RMSNorm
{
public:
    RMSNorm(ggml_type type, u32 dimensions, f32 epsilon = 1.0e-6);
    virtual ~RMSNorm();

private:
    s64 duration_;
    f32 epsilon_;
    Tensor weight_;
};

//--- Tokens
//-----------------------------------------------------------
class Tokens
{
public:
    struct String
    {
        u64 len_;
        const char8_t* str_;
    };

    Tokens();
    Tokens(const gguf::GGUF& model_data);
    ~Tokens();
    Tokens(Tokens&& other);
    Tokens& operator=(Tokens&& other);
private:
    Tokens(const Tokens&) = delete;
    Tokens& operator=(const Tokens&) = delete;
    u32 size_;
    HashMap<String, u32> tokenToId_;
    HashMap<u32, String> idToToken_;
};

//--- Tokenizer
//-----------------------------------------------------------
class GPT2TokenizerRef
{
public:
    struct String
    {
        u64 len_;
        const char8_t* str_;
    };

    GPT2TokenizerRef();
    GPT2TokenizerRef(const gguf::GGUF& model_data);
    ~GPT2TokenizerRef();
    GPT2TokenizerRef(GPT2TokenizerRef&& other);
    GPT2TokenizerRef& operator=(GPT2TokenizerRef&& other);
    String decode(int32_t token_id) const;
    Array<uint32_t> encode(const Array<char8_t>& text) const;

private:
    GPT2TokenizerRef(const GPT2TokenizerRef&) = delete;
    GPT2TokenizerRef& operator=(const GPT2TokenizerRef&) = delete;

    static const char8_t* Pattern;
    u32 size_;
    HashMap<String, u32> tokenToId_;
    HashMap<u32, String> idToToken_;
};

//--- Model
//-----------------------------------------------------------
class Model
{
public:
protected:
    Model();
    virtual ~Model();

private:
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
};

//--- Llama2
//-----------------------------------------------------------
class Llama2: public virtual Model
{
public:
protected:
    /*
    llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}
    */
    struct Config
    {
        uint32_t block_size_ = 2048;
        uint32_t vocab_size_ = 32000;
        uint32_t padding_vocab_size_ = 0;
        uint32_t n_layer_ = 32;
        uint32_t n_head_ = 32;
        uint32_t n_embd_ = 4096;
    };

    Llama2();
    explicit Llama2(const Config& config);
    Llama2(Llama2&& other);
    virtual ~Llama2();
    Llama2& operator=(Llama2&& other);

private:
    Llama2(const Llama2&) = delete;
    Llama2& operator=(const Llama2&) = delete;
    Config config_;
};
} // namespace cppgpt
#endif // INC_CPPGPT_H_

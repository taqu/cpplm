#ifndef INC_CPPGPT_H_
#define INC_CPPGPT_H_
#include "gguf.h"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <type_traits>
#include <initializer_list>
#include <memory>
#include <utility>
#include <re2/re2.h>

// new/delete
void* operator new(std::size_t size);
void* operator new(std::size_t size, std::align_val_t alignment);
void* operator new(std::size_t size, const std::nothrow_t&) noexcept;
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept;
//void* operator new(std::size_t size, void* ptr) noexcept;
void operator delete(void* ptr) noexcept;
void operator delete(void* ptr, std::size_t size) noexcept;
void operator delete(void* ptr, std::align_val_t alignment) noexcept;
void operator delete(void* ptr, std::size_t size, std::align_val_t alignment) noexcept;
void operator delete(void* ptr, const std::nothrow_t&) noexcept;
void operator delete(void* ptr, std::align_val_t alignment, const std::nothrow_t&) noexcept;
//void operator delete(void* ptr, void*) noexcept;

void* operator new[](std::size_t size);
void* operator new[](std::size_t size, std::align_val_t alignment);
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept;
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept;
//void* operator new[](std::size_t size, void* ptr) noexcept;
void operator delete[](void* ptr) noexcept;
void operator delete[](void* ptr, std::size_t size) noexcept;
void operator delete[](void* ptr, std::align_val_t alignment) noexcept;
void operator delete[](void* ptr, std::size_t size, std::align_val_t alignment) noexcept;
void operator delete[](void* ptr, const std::nothrow_t&) noexcept;
void operator delete[](void* ptr, std::align_val_t alignment, const std::nothrow_t&) noexcept;
//void operator delete[](void* ptr, void*) noexcept;

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
u32 next_prime(u32 x);

/**
@brief Find the first element in the range [first, last) which is more than equal val.
*/
template<class FwdIt, class T, class LessThan>
FwdIt lower_bound(FwdIt first, FwdIt last, const T& val, LessThan LT)
{
    typename std::iterator_traits<FwdIt>::difference_type count = last - first;
    while(0 < count) {
        typename std::iterator_traits<FwdIt>::difference_type d = count / 2;
        FwdIt m = first + d;
        if(LT(*m, val)) {
            first = m + 1;
            count -= d + 1;
        } else {
            count = d;
        }
    }
    return first;
}

template<class T>
struct Hasher
{
    u32 operator()(const T& x) const;
};

template<class T, class U>
class HashMap
{
    static_assert(std::is_trivially_copyable<T>::value == true, "T should be trivially copyable");
    static_assert(std::is_trivially_copyable<U>::value == true, "U should be trivially copyable");
public:
    inline static constexpr u32 Invalid = 0xFFFF'FFFFUL;

    HashMap();
    ~HashMap();
    u32 size() const;
    void clear();
    void add(const T& key, const U& value);
    void remove(const T& key);
    void remove(u32 pos);
    u32 find(const T& key) const;
    u32 end() const;
    void swap(HashMap& other);
private:
    HashMap(const HashMap&) = delete;
    HashMap& operator=(const HashMap&) = delete;

    inline static u64 align(u64 x)
    {
        return (x+7ULL)&(~7ULL);
    }

    explicit HashMap(u32 capacity);
    void expand();
    void create(u32 capacity);
    u32 find(const T& key, u32 hash) const;
    void remove(u32 pos, u32 hash);

    struct Entry
    {
        inline static constexpr u32 HashMask = 0x7FFFFFFFU;
        inline static constexpr u32 OccupyFlag = 0x80000000U;

        void clear()
        {
            hash_ = 0;
        }
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
u32 HashMap<T, U>::size() const
{
    return size_;
}

template<class T, class U>
void HashMap<T, U>::clear()
{
    for(u32 i=0; i<capacity_; ++i){
        if(entries_[i].isOccupy()){
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
    u32 hash = hasher_(key);
    remove(pos, hash);
}

template<class T, class U>
u32 HashMap<T, U>::find(const T& key) const
{
}

template<class T, class U>
u32 HashMap<T, U>::end() const
{
}

template<class T, class U>
void HashMap<T, U>::swap(HashMap& other)
{
}

template<class T, class U>
HashMap<T, U>::HashMap(u32 capacity)
    :capacity_(0)
    ,size_(0)
    ,empty_(Invalid)
    ,entries_(nullptr)
    ,keys_(nullptr)
    ,values_(nullptr)
{
    create(capacity);
}

template<class T, class U>
void HashMap<T, U>::expand()
{
    HashMap<T,U> tmp(capacity_+1);

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
    u64 entry_size = align(sizeof(Entry)*capacity_);
    u64 key_size = align(sizeof(T)*capacity_);
    u64 value_size = align(sizeof(U)*capacity_);
    u8* buffer = static_cast<u8*>(allocate(entry_size + key_size + value_size));
    Entry* entries = reinterpret_cast<Entry*>(buffer);
    T* keys = reinterpret_cast<T*>(buffer + entry_size);
    U* values = reinterpret_cast<U*>(buffer + entry_size + key_size);
    for(u32 i=0; i<capacity_; ++i){
        entries[i].index_ = Invalid;
        entries[i].next_ = i+1;
        entries[i].hash_ = 0;
    }
    entries[capacity_-1].next_ = Invalid;
    empty_ = 0;
    deallocate(entries_);
    entries_ = entries;
    keys_ = keys;
    values_ = values;
}

template<class T, class U>
u32 HashMap<T, U>::find(const T& key, u32 hash) const
{
    assert(0<capacity_);
    u32 pos = hash % capacity_;
    hash |= Entry::OccupyFlag;
    do{
        if(!entries_[pos].isOccupy()){
            break;
        }
        if(hash == entries_[pos].hash_ && key==keys_[pos]){
            return i;
        }
        pos = entries_[pos].next_;
    }while(pos != Invalid);
    return end();
}

template<class T, class U>
void HashMap<T,U>::remove(u32 pos, u32 hash)
{
    u32 entryPos = hash % capacity_;

    keys_[pos].~T();
    values_[pos].~U();

    if(pos == entries_[entryPos].index_) {
        entries_[entryPos].index_ = entries_[pos].next_;
    } else {
        for(u32 p = entries_[entryPos].index_; entries_[p].next_ != pos; p = entries_[p].next_) {
#if _DEBUG
            if(buckets_[p].next_ == Invalid) {
                assert(false);
            }
#endif
        }
        buckets_[p].next_ = buckets_[pos].next_;
    }
    buckets_[pos].next_ = freeList_;
    freeList_ = pos;
    buckets_[pos].clear();
    --size_;
}

// available tensor operations
enum class ggml_op
{
    GGML_OP_NONE = 0,

    GGML_OP_DUP,
    GGML_OP_ADD,
    GGML_OP_ADD1,
    GGML_OP_ACC,
    GGML_OP_SUB,
    GGML_OP_MUL,
    GGML_OP_DIV,
    GGML_OP_SQR,
    GGML_OP_SQRT,
    GGML_OP_LOG,
    GGML_OP_SIN,
    GGML_OP_COS,
    GGML_OP_SUM,
    GGML_OP_SUM_ROWS,
    GGML_OP_MEAN,
    GGML_OP_ARGMAX,
    GGML_OP_REPEAT,
    GGML_OP_REPEAT_BACK,
    GGML_OP_CONCAT,
    GGML_OP_SILU_BACK,
    GGML_OP_NORM, // normalize
    GGML_OP_RMS_NORM,
    GGML_OP_RMS_NORM_BACK,
    GGML_OP_GROUP_NORM,

    GGML_OP_MUL_MAT,
    GGML_OP_MUL_MAT_ID,
    GGML_OP_OUT_PROD,

    GGML_OP_SCALE,
    GGML_OP_SET,
    GGML_OP_CPY,
    GGML_OP_CONT,
    GGML_OP_RESHAPE,
    GGML_OP_VIEW,
    GGML_OP_PERMUTE,
    GGML_OP_TRANSPOSE,
    GGML_OP_GET_ROWS,
    GGML_OP_GET_ROWS_BACK,
    GGML_OP_DIAG,
    GGML_OP_DIAG_MASK_INF,
    GGML_OP_DIAG_MASK_ZERO,
    GGML_OP_SOFT_MAX,
    GGML_OP_SOFT_MAX_BACK,
    GGML_OP_ROPE,
    GGML_OP_ROPE_BACK,
    GGML_OP_CLAMP,
    GGML_OP_CONV_TRANSPOSE_1D,
    GGML_OP_IM2COL,
    GGML_OP_IM2COL_BACK,
    GGML_OP_CONV_TRANSPOSE_2D,
    GGML_OP_POOL_1D,
    GGML_OP_POOL_2D,
    GGML_OP_POOL_2D_BACK,
    GGML_OP_UPSCALE, // nearest interpolate
    GGML_OP_PAD,
    GGML_OP_ARANGE,
    GGML_OP_TIMESTEP_EMBEDDING,
    GGML_OP_ARGSORT,
    GGML_OP_LEAKY_RELU,

    GGML_OP_FLASH_ATTN_EXT,
    GGML_OP_FLASH_ATTN_BACK,
    GGML_OP_SSM_CONV,
    GGML_OP_SSM_SCAN,
    GGML_OP_WIN_PART,
    GGML_OP_WIN_UNPART,
    GGML_OP_GET_REL_POS,
    GGML_OP_ADD_REL_POS,
    GGML_OP_RWKV_WKV,

    GGML_OP_UNARY,

    GGML_OP_MAP_UNARY,
    GGML_OP_MAP_BINARY,

    GGML_OP_MAP_CUSTOM1_F32,
    GGML_OP_MAP_CUSTOM2_F32,
    GGML_OP_MAP_CUSTOM3_F32,

    GGML_OP_MAP_CUSTOM1,
    GGML_OP_MAP_CUSTOM2,
    GGML_OP_MAP_CUSTOM3,

    GGML_OP_CROSS_ENTROPY_LOSS,
    GGML_OP_CROSS_ENTROPY_LOSS_BACK,

    GGML_OP_COUNT,
};

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

    inline s64 time() const{ return duration_;}
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

    inline s64 time() const{ return duration_;}
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

    inline s64 time() const{ return duration_;}
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

    inline s64 time() const{ return duration_;}
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

    inline s64 time() const{ return duration_;}
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

    inline s64 time() const{ return duration_;}
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

    inline s64 time() const{ return duration_;}
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

//--- Tokenizer
//-----------------------------------------------------------
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

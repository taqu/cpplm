#ifndef INC_CPPGPT_H_
#define INC_CPPGPT_H_
#include "gguf.h"
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <memory>

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

void* allocate(size_t size, size_t align=16);
void deallocate(void* ptr, size_t align=16);

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
}

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
    Tensor(ggml_type type, std::initializer_list<u64> dimensions, void* data);
    Tensor(ggml_type type, const Tensor& shape);
    Tensor(Tensor&& other);
    ~Tensor();
    Tensor& operator=(Tensor&& other);
    ggml_type type() const;
    u32 num_dims() const;
    u64 total_size() const;
    u64 total_bytes() const;
    u64 size(u32 index) const;

    template <typename T>
    const T* data() const
    { 
        return reinterpret_cast<const T*>(data_.get()); 
    }

    template <typename T>
    T* data()
    { 
        return reinterpret_cast<T*>(data_.get()); 
    }

    void resize(std::initializer_list<u64> dimensions) noexcept;
private:
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    struct CustomDeleter
    {
        constexpr CustomDeleter(bool dummy) noexcept
            :dummy_(dummy)
        {
        }

        void operator()(u8* ptr) const
        {
            if(dummy_){
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
    Tensor&& convertF32(const Tensor& input);
    f32 kahan_sum(u64 size, const f32* src);
    f32 kahan_sum_squared(u64 size, const f32* src, f32 mean);
    void normalize_vec(u64 size, f32* dst, const f32* src, const f32* weight, const f32* bias);
    Tensor&& normalize(const Tensor& input, const Tensor& weight, const Tensor& bias);
}

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
    Tensor&& forward(const Tensor& input);

private:
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
    Tensor&& forward(const Tensor& input);
    Tensor&& forward_proj(const Tensor& input);

private:
    Tensor weight_;
};

//--- RMSNorm
//-----------------------------------------------------------
class RMSNorm: public Module
{
public:
    RMSNorm(ggml_type type, u32 dimensions, f32 epsilon = 1.0e-6);
    virtual ~RMSNorm();

private:
    f32 epsilon_;
    Tensor weight_;
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

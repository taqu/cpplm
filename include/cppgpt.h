#ifndef INC_CPPGPT_H_
#define INC_CPPGPT_H_
#include "gguf.h"
#include <cassert>
#include <cstdint>

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

//--- MemoryView
//-----------------------------------------------------------
class MemoryView
{
public:
    MemoryView();
    explicit MemoryView(ggml_type type, u64 size, const void* ptr);
    MemoryView(MemoryView&& other);
    ~MemoryView();
    MemoryView& operator=(MemoryView&& other);

    u64 size() const;

    u32 operator[](u64 i) const;

private:
    MemoryView(const MemoryView&) = delete;
    MemoryView& operator=(const MemoryView&) = delete;
    ggml_type type_;
    u32 bits_;
    u64 size_;
    const void* ptr_;
};

//--- Tensor
//-----------------------------------------------------------
class Tensor
{
public:
    Tensor();
    Tensor(ggml_type type, u64 dimensions);
    ~Tensor();
    void zeros();
    void ones();

private:
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    ggml_type type_;
    Memory data_;
};

//--- Module
//-----------------------------------------------------------
class Module
{
public:
    virtual ~Module()
    {
    }
    virtual dnnl::memory forward(dnnl::memory tensor) = 0;

protected:
    Module()
    {
    }

private:
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
};

//--- RMSNorm
//-----------------------------------------------------------
class RMSNorm: public Module
{
public:
    RMSNorm(ggml_type type, u32 dimensions, f32 epsilon = 1.0e-6);
    virtual ~RMSNorm();
    virtual dnnl::memory forward(dnnl::memory tensor) override;

private:
    dnnl::memory norm(dnnl::memory x);
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
    std::vector<dnnl::primitive> network_;
    std::vector<std::unordered_map<int32_t, dnnl::memory>> args_;
};
} // namespace cppgpt
#endif // INC_CPPGPT_H_

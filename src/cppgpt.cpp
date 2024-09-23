#include "cppgpt.h"
#include <mimalloc-2.1/mimalloc.h>

namespace cppgpt
{
namespace
{
    u32 get_bit_size(ggml_type type)
    {
        switch(type) {
        case ggml_type::GGML_TYPE_F32:
            return 32;
        case ggml_type::GGML_TYPE_F16:
            return 16;
        case ggml_type::GGML_TYPE_Q4_0:
            return 4;
        case ggml_type::GGML_TYPE_Q4_1:
            return 4;
        case ggml_type::GGML_TYPE_Q5_0:
            return 5;
        case ggml_type::GGML_TYPE_Q5_1:
            return 5;
        case ggml_type::GGML_TYPE_Q8_0:
            return 8;
        case ggml_type::GGML_TYPE_Q8_1:
            return 8;
        case ggml_type::GGML_TYPE_Q2_K:
            return 2;
        case ggml_type::GGML_TYPE_Q3_K:
            return 3;
        case ggml_type::GGML_TYPE_Q4_K:
            return 4;
        case ggml_type::GGML_TYPE_Q5_K:
            return 5;
        case ggml_type::GGML_TYPE_Q6_K:
            return 6;
        case ggml_type::GGML_TYPE_Q8_K:
            return 8;
        case ggml_type::GGML_TYPE_IQ2_XXS:
            return 2;
        case ggml_type::GGML_TYPE_IQ2_XS:
            return 2;
        case ggml_type::GGML_TYPE_IQ3_XXS:
            return 3;
        case ggml_type::GGML_TYPE_IQ1_S:
            return 1;
        case ggml_type::GGML_TYPE_IQ4_NL:
            return 4;
        case ggml_type::GGML_TYPE_IQ3_S:
            return 3;
        case ggml_type::GGML_TYPE_IQ2_S:
            return 2;
        case ggml_type::GGML_TYPE_IQ4_XS:
            return 4;
        case ggml_type::GGML_TYPE_I8:
            return 8;
        case ggml_type::GGML_TYPE_I16:
            return 16;
        case ggml_type::GGML_TYPE_I32:
            return 32;
        case ggml_type::GGML_TYPE_I64:
            return 64;
        case ggml_type::GGML_TYPE_F64:
            return 64;
        case ggml_type::GGML_TYPE_IQ1_M:
            return 1;
        default:
            assert(false);
            return 0;
        }
    }

    u32 get_bit_size_aligned(ggml_type type)
    {
        switch(type) {
        case ggml_type::GGML_TYPE_F32:
            return 32;
        case ggml_type::GGML_TYPE_F16:
            return 16;
        case ggml_type::GGML_TYPE_Q4_0:
            return 8;
        case ggml_type::GGML_TYPE_Q4_1:
            return 8;
        case ggml_type::GGML_TYPE_Q5_0:
            return 8;
        case ggml_type::GGML_TYPE_Q5_1:
            return 8;
        case ggml_type::GGML_TYPE_Q8_0:
            return 8;
        case ggml_type::GGML_TYPE_Q8_1:
            return 8;
        case ggml_type::GGML_TYPE_Q2_K:
            return 8;
        case ggml_type::GGML_TYPE_Q3_K:
            return 8;
        case ggml_type::GGML_TYPE_Q4_K:
            return 8;
        case ggml_type::GGML_TYPE_Q5_K:
            return 8;
        case ggml_type::GGML_TYPE_Q6_K:
            return 8;
        case ggml_type::GGML_TYPE_Q8_K:
            return 8;
        case ggml_type::GGML_TYPE_IQ2_XXS:
            return 8;
        case ggml_type::GGML_TYPE_IQ2_XS:
            return 8;
        case ggml_type::GGML_TYPE_IQ3_XXS:
            return 8;
        case ggml_type::GGML_TYPE_IQ1_S:
            return 8;
        case ggml_type::GGML_TYPE_IQ4_NL:
            return 8;
        case ggml_type::GGML_TYPE_IQ3_S:
            return 8;
        case ggml_type::GGML_TYPE_IQ2_S:
            return 8;
        case ggml_type::GGML_TYPE_IQ4_XS:
            return 8;
        case ggml_type::GGML_TYPE_I8:
            return 8;
        case ggml_type::GGML_TYPE_I16:
            return 16;
        case ggml_type::GGML_TYPE_I32:
            return 32;
        case ggml_type::GGML_TYPE_I64:
            return 64;
        case ggml_type::GGML_TYPE_F64:
            return 64;
        case ggml_type::GGML_TYPE_IQ1_M:
            return 8;
        default:
            assert(false);
            return 0;
        }
    }

    u32 get_byte_size_aligned(ggml_type type)
    {
        switch(type) {
        case ggml_type::GGML_TYPE_F32:
            return 4;
        case ggml_type::GGML_TYPE_F16:
            return 2;
        case ggml_type::GGML_TYPE_Q4_0:
            return 1;
        case ggml_type::GGML_TYPE_Q4_1:
            return 1;
        case ggml_type::GGML_TYPE_Q5_0:
            return 1;
        case ggml_type::GGML_TYPE_Q5_1:
            return 1;
        case ggml_type::GGML_TYPE_Q8_0:
            return 1;
        case ggml_type::GGML_TYPE_Q8_1:
            return 1;
        case ggml_type::GGML_TYPE_Q2_K:
            return 1;
        case ggml_type::GGML_TYPE_Q3_K:
            return 1;
        case ggml_type::GGML_TYPE_Q4_K:
            return 1;
        case ggml_type::GGML_TYPE_Q5_K:
            return 1;
        case ggml_type::GGML_TYPE_Q6_K:
            return 1;
        case ggml_type::GGML_TYPE_Q8_K:
            return 1;
        case ggml_type::GGML_TYPE_IQ2_XXS:
            return 1;
        case ggml_type::GGML_TYPE_IQ2_XS:
            return 1;
        case ggml_type::GGML_TYPE_IQ3_XXS:
            return 1;
        case ggml_type::GGML_TYPE_IQ1_S:
            return 1;
        case ggml_type::GGML_TYPE_IQ4_NL:
            return 1;
        case ggml_type::GGML_TYPE_IQ3_S:
            return 1;
        case ggml_type::GGML_TYPE_IQ2_S:
            return 1;
        case ggml_type::GGML_TYPE_IQ4_XS:
            return 1;
        case ggml_type::GGML_TYPE_I8:
            return 1;
        case ggml_type::GGML_TYPE_I16:
            return 2;
        case ggml_type::GGML_TYPE_I32:
            return 4;
        case ggml_type::GGML_TYPE_I64:
            return 8;
        case ggml_type::GGML_TYPE_F64:
            return 8;
        case ggml_type::GGML_TYPE_IQ1_M:
            return 1;
        default:
            assert(false);
            return 0;
        }
    }

    u32 get(u32 bits, void* data)
    {
    }
} // namespace

void* allocate(size_t size, size_t align)
{
    return mi_malloc_aligned(size, align);
}

void deallocate(void* ptr, size_t align)
{
    mi_free_aligned(ptr, align);
}

//--- Memory
//-----------------------------------------------------------

Memory::Memory()
    :size_(0)
    ,ptr_(nullptr)
{
}

Memory::Memory(Memory&& other)
    :size_(other.size_)
    ,ptr_(other.ptr_)
{
    other.size_ = 0;
    other.ptr_ = nullptr;
}

Memory::Memory(u64 size)
    :size_(size)
    ,ptr_(nullptr)
{
}

Memory::~Memory()
{
    deallocate(ptr_);
    ptr_ = nullptr;
}

Memory& Memory::operator=(Memory&& other)
{
    if(this != &other){
        deallocate(ptr_);
        size_ = other.size_;
        ptr_ = other.ptr_;
        other.size_ = 0;
        other.ptr_ = nullptr;
    }
    return *this;
}

//--- MemoryView
//-----------------------------------------------------------
MemoryView::MemoryView()
    :type_(ggml_type::GGML_TYPE_F32)
    ,bits_(0)
    ,size_(0)
    ,ptr_(nullptr)
{
}

    MemoryView::MemoryView(ggml_type type, u64 size, const void* ptr)
    :type_(type)
        ,bits_(get_bit_size(type))
        ,size_(size)
        ,ptr_(ptr)
{
        assert(nullptr != ptr);
}

    MemoryView::MemoryView(MemoryView&& other)
{
}

    MemoryView::~MemoryView()
{
}

    MemoryView& MemoryView::operator=(MemoryView&& other)
{
        if(this != &other){
            type_ = other.type_;
            bits_ = other.bits_;
            size_ = other.size_;
            ptr_ = other.ptr_;
        }
        return *this;
}

    u64 MemoryView::size() const
{
        return size_;
}

u32 MemoryView::operator[](u64 i) const
{
    assert(i < size_);
    u64 total_bits = bits_ * size_;
    switch(type_) {
    case ggml_type::GGML_TYPE_F32:
        return 4;
    case ggml_type::GGML_TYPE_F16:
        return 2;
    case ggml_type::GGML_TYPE_Q4_0:
        return 1;
    case ggml_type::GGML_TYPE_Q4_1:
        return 1;
    case ggml_type::GGML_TYPE_Q5_0:
        return 1;
    case ggml_type::GGML_TYPE_Q5_1:
        return 1;
    case ggml_type::GGML_TYPE_Q8_0:
        return 1;
    case ggml_type::GGML_TYPE_Q8_1:
        return 1;
    case ggml_type::GGML_TYPE_Q2_K:
        return 1;
    case ggml_type::GGML_TYPE_Q3_K:
        return 1;
    case ggml_type::GGML_TYPE_Q4_K:
        return 1;
    case ggml_type::GGML_TYPE_Q5_K:
        return 1;
    case ggml_type::GGML_TYPE_Q6_K:
        return 1;
    case ggml_type::GGML_TYPE_Q8_K:
        return 1;
    case ggml_type::GGML_TYPE_IQ2_XXS:
        return 1;
    case ggml_type::GGML_TYPE_IQ2_XS:
        return 1;
    case ggml_type::GGML_TYPE_IQ3_XXS:
        return 1;
    case ggml_type::GGML_TYPE_IQ1_S:
        return 1;
    case ggml_type::GGML_TYPE_IQ4_NL:
        return 1;
    case ggml_type::GGML_TYPE_IQ3_S:
        return 1;
    case ggml_type::GGML_TYPE_IQ2_S:
        return 1;
    case ggml_type::GGML_TYPE_IQ4_XS:
        return 1;
    case ggml_type::GGML_TYPE_I8:
        return 1;
    case ggml_type::GGML_TYPE_I16:
        return 2;
    case ggml_type::GGML_TYPE_I32:
        return 4;
    case ggml_type::GGML_TYPE_I64:
        return 8;
    case ggml_type::GGML_TYPE_F64:
        return 8;
    case ggml_type::GGML_TYPE_IQ1_M:
        return 1;
    default:
        assert(false);
        return 0;
    }
}

//--- Tensor
//-----------------------------------------------------------
namespace
{
#if 0
    void write_to_dnnl_memory(dnnl::memory& mem, void* src)
    {
        assert(mem);
        assert(nullptr != src);
        dnnl::engine eng = mem.get_engine();
        size_t size = mem.get_desc().get_size();
#ifdef DNNL_WITH_SYCL
        bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
                            && eng.get_kind() == dnnl::engine::kind::cpu);
        bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
                            && eng.get_kind() == dnnl::engine::kind::gpu);
        if(is_cpu_sycl || is_gpu_sycl) {
            auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
            if(mkind == dnnl::sycl_interop::memory_kind::buffer) {
                auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
                auto dst = buffer.get_host_access();
                uint8_t* dst_ptr = dst.get_pointer();
                if(!dst_ptr)
                    throw std::runtime_error("get_pointer returned nullptr.");
                for(size_t i = 0; i < size; ++i)
                    dst_ptr[i] = ((uint8_t*)handle)[i];
            } else {
                assert(mkind == dnnl::sycl_interop::memory_kind::usm);
                uint8_t* dst_ptr = (uint8_t*)mem.get_data_handle();
                if(!dst_ptr)
                    throw std::runtime_error("get_data_handle returned nullptr.");
                if(is_cpu_sycl) {
                    for(size_t i = 0; i < size; ++i)
                        dst_ptr[i] = ((uint8_t*)handle)[i];
                } else {
                    auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                    sycl_queue.memcpy(dst_ptr, handle, size).wait();
                }
            }
            return;
        }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if(eng.get_kind() == dnnl::engine::kind::gpu) {
            void* mapped_ptr = mem.map_data();
            if(nullptr != mapped_ptr) {
                ::memcpy(mapped_ptr, src, size);
            }
            mem.unmap_data(mapped_ptr);
            return;
        }
#endif

        if(eng.get_kind() == dnnl::engine::kind::cpu) {
            uint8_t* dst = static_cast<uint8_t*>(mem.get_data_handle());
            if(nullptr != dst) {
                ::memcpy(dst, src, size);
            }
            return;
        }
    }

    void memset_dnnl_memory(dnnl::memory& mem, int32_t x)
    {
        assert(mem);
        dnnl::engine eng = mem.get_engine();
        size_t size = mem.get_desc().get_size();
#ifdef DNNL_WITH_SYCL
        bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
                            && eng.get_kind() == dnnl::engine::kind::cpu);
        bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
                            && eng.get_kind() == dnnl::engine::kind::gpu);
        if(is_cpu_sycl || is_gpu_sycl) {
            auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
            if(mkind == dnnl::sycl_interop::memory_kind::buffer) {
                auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
                auto dst = buffer.get_host_access();
                uint8_t* dst_ptr = dst.get_pointer();
                if(!dst_ptr)
                    throw std::runtime_error("get_pointer returned nullptr.");
                for(size_t i = 0; i < size; ++i)
                    dst_ptr[i] = ((uint8_t*)handle)[i];
            } else {
                assert(mkind == dnnl::sycl_interop::memory_kind::usm);
                uint8_t* dst_ptr = (uint8_t*)mem.get_data_handle();
                if(!dst_ptr)
                    throw std::runtime_error("get_data_handle returned nullptr.");
                if(is_cpu_sycl) {
                    for(size_t i = 0; i < size; ++i)
                        dst_ptr[i] = ((uint8_t*)handle)[i];
                } else {
                    auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                    sycl_queue.memcpy(dst_ptr, handle, size).wait();
                }
            }
            return;
        }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if(eng.get_kind() == dnnl::engine::kind::gpu) {
            void* mapped_ptr = mem.map_data();
            if(nullptr != mapped_ptr) {
                ::memset(mapped_ptr, x, size);
            }
            mem.unmap_data(mapped_ptr);
            return;
        }
#endif

        if(eng.get_kind() == dnnl::engine::kind::cpu) {
            uint8_t* dst = static_cast<uint8_t*>(mem.get_data_handle());
            if(nullptr != dst) {
                ::memset(dst, 0, size);
            }
            return;
        }
    }
#endif
} // namespace

Tensor::Tensor()
    : type_(ggml_type::GGML_TYPE_F32)
{
}

Tensor::Tensor(ggml_type type, u64 dimensions)
    : type_(type)
    , data_(get_byte_size_aligned(type)*dimensions)
{
}

Tensor::~Tensor()
{
}

void Tensor::zeros()
{
    memset_dnnl_memory(data_, 0);
}

void Tensor::ones()
{
    memset_dnnl_memory(data_, 1);
}

//--- RMSNorm
//-----------------------------------------------------------
RMSNorm::RMSNorm(ggml_type type, u32 dimensions, f32 epsilon)
    : epsilon_(epsilon)
    ,weight_(type, {dimensions})
{
}

RMSNorm::~RMSNorm()
{
}

dnnl::memory RMSNorm::forward(dnnl::memory tensor)
{
}

dnnl::memory RMSNorm::norm(dnnl::memory x)
{
}

//--- Model
//-----------------------------------------------------------
Model::Model()
{
}

Model::~Model()
{
}

//--- Llama2
//-----------------------------------------------------------
Llama2::Llama2()
{
}

Llama2::Llama2(const Config& config)
    : config_(config)
{
}

Llama2::Llama2(Llama2&& other)
    : network_(std::move(other.network_))
{
}

Llama2::~Llama2()
{
    network_.clear();
}

Llama2& Llama2::operator=(Llama2&& other)
{
    if(this == &other) {
        return *this;
    }
    config_ = other.config_;
    network_ = std::move(other.network_);
    return *this;
}
} // namespace cppgpt

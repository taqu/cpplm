#include "cppgpt.h"
#include <cmath>
#include <functional>
#include <immintrin.h>
#include <mimalloc-2.1/mimalloc.h>

// new/delete
void* operator new(std::size_t size)
{
    return cppgpt::allocate(size);
}

void* operator new(std::size_t size, std::align_val_t alignment)
{
    return cppgpt::allocate(size, static_cast<std::size_t>(alignment));
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept
{
    return cppgpt::allocate(size);
}

void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return cppgpt::allocate(size, static_cast<std::size_t>(alignment));
}

//void* operator new(std::size_t /*size*/, void* ptr) noexcept
//{
//    return ptr;
//}

void operator delete(void* ptr) noexcept
{
    cppgpt::deallocate(ptr);
}

void operator delete(void* ptr, std::size_t /*size*/) noexcept
{
    cppgpt::deallocate(ptr);
}

void operator delete(void* ptr, std::align_val_t alignment) noexcept
{
    cppgpt::deallocate(ptr, static_cast<std::size_t>(alignment));
}

void operator delete(void* ptr, std::size_t /*size*/, std::align_val_t alignment) noexcept
{
    cppgpt::deallocate(ptr, static_cast<std::size_t>(alignment));
}

void operator delete(void* ptr, const std::nothrow_t&) noexcept
{
    cppgpt::deallocate(ptr);
}

void operator delete(void* ptr, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    cppgpt::deallocate(ptr, static_cast<std::size_t>(alignment));
}

//void operator delete(void* ptr, void*) noexcept
//{
//    (void)ptr;
//}

void* operator new[](std::size_t size)
{
    return cppgpt::allocate(size);
}

void* operator new[](std::size_t size, std::align_val_t alignment)
{
    return cppgpt::allocate(size, static_cast<std::size_t>(alignment));
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept
{
    return cppgpt::allocate(size);
}

void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    return cppgpt::allocate(size, static_cast<std::size_t>(alignment));
}

//void* operator new[](std::size_t /*size*/, void* ptr) noexcept
//{
//    return ptr;
//}

void operator delete[](void* ptr) noexcept
{
    cppgpt::deallocate(ptr);
}

void operator delete[](void* ptr, std::size_t /*size*/) noexcept
{
    cppgpt::deallocate(ptr);
}

void operator delete[](void* ptr, std::align_val_t alignment) noexcept
{
    cppgpt::deallocate(ptr, static_cast<std::size_t>(alignment));
}

void operator delete[](void* ptr, std::size_t /*size*/, std::align_val_t alignment) noexcept
{
    cppgpt::deallocate(ptr, static_cast<std::size_t>(alignment));
}

void operator delete[](void* ptr, const std::nothrow_t&) noexcept
{
    cppgpt::deallocate(ptr);
}

void operator delete[](void* ptr, std::align_val_t alignment, const std::nothrow_t&) noexcept
{
    cppgpt::deallocate(ptr, static_cast<std::size_t>(alignment));
}

//void operator delete[](void* ptr, void*) noexcept
//{
//    (void)ptr;
//}

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
} // namespace

namespace util
{
    void copy1(u32 bits, u64 size, void* dst, const void* src)
    {
        assert(1 == bits);
        (void)bits;
        u8* dstu8 = static_cast<u8*>(dst);
        const u8* srcu8 = static_cast<const u8*>(src);
        for(u64 i = 0; i < size; ++i) {
            u64 index = i >> 3;
            u64 bit = i - (index << 3);
            u8 x = 0 == (srcu8[index] & bit) ? 0 : 255;
            dstu8[i] = x;
        }
    }

    void copy2(u32 bits, u64 size, void* dst, const void* src)
    {
        assert(2 == bits);
        (void)bits;
        static const u8 values[4] = {
            0,
            85,
            170,
            255,
        };
        u8* dstu8 = static_cast<u8*>(dst);
        const u8* srcu8 = static_cast<const u8*>(src);
        for(u64 i = 0; i < size; ++i) {
            u64 index = i >> 2;
            u64 bit = (srcu8[index] >> ((i - (index << 2)) << 1)) & 0x3UL;
            assert(bit < 4);
            dstu8[i] = values[bit];
        }
    }

    void copy3(u32 bits, u64 size, void* dst, const void* src)
    {
        assert(3 == bits);
        (void)bits;
        static const u8 values[8] = {
            0,
            37,
            73,
            110,
            146,
            183,
            219,
            255,
        };
        u8* dstu8 = static_cast<u8*>(dst);
        const u8* srcu8 = static_cast<const u8*>(src);
        for(u64 i = 0; i < size; ++i) {
            u64 bit_index = i * 3;
            u64 index = bit_index >> 3;
            u64 remain = bit_index - (index << 3);
            u64 bit = (srcu8[index] >> remain) & 0x7UL;
            assert(remain < 8);
            if(6 <= remain) {
                switch(8 - remain) {
                case 1:
                    bit = bit | ((srcu8[index + 1] & 0x1UL) << 2UL);
                    break;
                case 2:
                    bit = bit | ((srcu8[index + 1] & 0x3UL) << 1UL);
                    break;
                }
            }
            assert(bit < 8);
            dstu8[i] = values[bit];
        }
    }

    void copy4(u32 bits, u64 size, void* dst, const void* src)
    {
        assert(4 == bits);
        (void)bits;
        // clang-format off
        static const u8 values[16] = {
            0, 17, 34, 51, 68, 85, 102, 119,
            137, 154, 171, 188, 205, 222, 239, 255,
        };
        // clang-format on
        u8* dstu8 = static_cast<u8*>(dst);
        const u8* srcu8 = static_cast<const u8*>(src);
        for(u64 i = 0; i < size; ++i) {
            u64 bit_index = i << 2;
            u64 index = bit_index >> 3;
            u64 remain = bit_index - (index << 3);
            u64 bit = (srcu8[index] >> remain) & 0x15UL;
            assert(remain < 8);
            if(5 <= remain) {
                switch(8 - remain) {
                case 1:
                    bit = bit | ((srcu8[index + 1] & 0x1UL) << 3UL);
                    break;
                case 2:
                    bit = bit | ((srcu8[index + 1] & 0x3UL) << 2UL);
                    break;
                case 3:
                    bit = bit | ((srcu8[index + 1] & 0x7UL) << 1UL);
                    break;
                }
            }
            assert(bit < 16);
            dstu8[i] = values[bit];
        }
    }

    void copy5(u32 bits, u64 size, void* dst, const void* src)
    {
        assert(5 == bits);
        (void)bits;
        // clang-format off
        static const u8 values[32] = {
            0, 8, 17, 25, 33, 41, 50, 58,
            66, 74, 83, 91, 99, 107, 116, 124,
            132, 140, 149, 157, 165, 173, 182, 190,
            198, 206, 215, 223, 231, 239, 248, 255,
        };
        // clang-format on
        u8* dstu8 = static_cast<u8*>(dst);
        const u8* srcu8 = static_cast<const u8*>(src);
        for(u64 i = 0; i < size; ++i) {
            u64 bit_index = (i << 2) + i;
            u64 index = bit_index >> 3;
            u64 remain = bit_index - (index << 3);
            u64 bit = (srcu8[index] >> remain) & 0x31UL;
            assert(remain < 8);
            if(4 <= remain) {
                switch(8 - remain) {
                case 1:
                    bit = bit | ((srcu8[index + 1] & 0x1UL) << 4UL);
                    break;
                case 2:
                    bit = bit | ((srcu8[index + 1] & 0x3UL) << 3UL);
                    break;
                case 3:
                    bit = bit | ((srcu8[index + 1] & 0x7UL) << 2UL);
                    break;
                case 4:
                    bit = bit | ((srcu8[index + 1] & 0xFUL) << 1UL);
                    break;
                }
            }
            assert(bit < 32);
            dstu8[i] = values[bit];
        }
    }

    void copy6(u32 bits, u64 size, void* dst, const void* src)
    {
        assert(6 == bits);
        (void)bits;
        // clang-format off
        static const u8 values[64] = {
            0, 4, 8, 12, 16, 20, 24, 28,
            33, 37, 41, 45, 49, 53, 57, 61,
            65, 69, 73, 77, 81, 85, 89, 93,
            98, 102, 106, 110, 114, 118, 122, 126,
            130, 134, 138, 142, 146, 150, 154, 158,
            163, 167, 171, 175, 179, 183, 187, 191,
            195, 199, 203, 207, 211, 215, 219, 223,
            228, 232, 236, 240, 244, 248, 252, 255,
        };
        // clang-format on
        u8* dstu8 = static_cast<u8*>(dst);
        const u8* srcu8 = static_cast<const u8*>(src);
        for(u64 i = 0; i < size; ++i) {
            u64 bit_index = i * 6;
            u64 index = bit_index >> 3;
            u64 remain = bit_index - (index << 3);
            u64 bit = (srcu8[index] >> remain) & 0x63UL;
            assert(remain < 8);
            if(3 <= remain) {
                switch(8 - remain) {
                case 1:
                    bit = bit | ((srcu8[index + 1] & 0x1UL) << 5UL);
                    break;
                case 2:
                    bit = bit | ((srcu8[index + 1] & 0x3UL) << 4UL);
                    break;
                case 3:
                    bit = bit | ((srcu8[index + 1] & 0x7UL) << 3UL);
                    break;
                case 4:
                    bit = bit | ((srcu8[index + 1] & 0xFUL) << 2UL);
                    break;
                case 5:
                    bit = bit | ((srcu8[index + 1] & 0x1FUL) << 1UL);
                    break;
                }
            }
            assert(bit < 64);
            dstu8[i] = values[bit];
        }
    }

    void copy8(u32 bits, u64 size, void* dst, const void* src)
    {
        assert(8 == bits);
        (void)bits;
        ::memcpy(dst, src, size * sizeof(u8));
    }

    void copy16(u32 bits, u64 size, void* dst, const void* src)
    {
        assert(16 == bits);
        (void)bits;
        ::memcpy(dst, src, size * sizeof(u16));
    }

    void copy32(u32 bits, u64 size, void* dst, const void* src)
    {
        assert(32 == bits);
        (void)bits;
        ::memcpy(dst, src, size * sizeof(u32));
    }

    void copy64(u32 bits, u64 size, void* dst, const void* src)
    {
        assert(64 == bits);
        (void)bits;
        ::memcpy(dst, src, size * sizeof(u64));
    }

    void copy1_f(u64 size, void* dst, const void* src)
    {
        f32* dstf = static_cast<f32*>(dst);
        const u8* srcu8 = static_cast<const u8*>(src);
        for(u64 i = 0; i < size; ++i) {
            u64 index = i >> 3;
            u64 bit = i - (index << 3);
            f32 x = 0 == (srcu8[index] & bit) ? 0.0f : 1.0f;
            dstf[i] = x;
        }
    }

    void copy2_f(u64 size, void* dst, const void* src)
    {
        static const f32 values[4] = {
            0.0f,
            0.333333f,
            0.666667f,
            1.0f,
        };
        f32* dstf = static_cast<f32*>(dst);
        const u8* srcu8 = static_cast<const u8*>(src);
        for(u64 i = 0; i < size; ++i) {
            u64 index = i >> 2;
            u64 bit = (srcu8[index] >> ((i - (index << 2)) << 1)) & 0x3UL;
            assert(bit < 4);
            dstf[i] = values[bit];
        }
    }

    void copy3_f(u64 size, void* dst, const void* src)
    {
        static const f32 values[8] = {
            0.0f,
            0.142857f,
            0.285714f,
            0.428571f,
            0.571429f,
            0.714286f,
            0.857143f,
            1.0f,
        };
        f32* dstf = static_cast<f32*>(dst);
        const u8* srcu8 = static_cast<const u8*>(src);
        for(u64 i = 0; i < size; ++i) {
            u64 bit_index = i * 3;
            u64 index = bit_index >> 3;
            u64 remain = bit_index - (index << 3);
            u64 bit = (srcu8[index] >> remain) & 0x7UL;
            assert(remain < 8);
            if(6 <= remain) {
                switch(8 - remain) {
                case 1:
                    bit = bit | ((srcu8[index + 1] & 0x1UL) << 2UL);
                    break;
                case 2:
                    bit = bit | ((srcu8[index + 1] & 0x3UL) << 1UL);
                    break;
                }
            }
            assert(bit < 8);
            dstf[i] = values[bit];
        }
    }

    void copy4_f(u64 size, void* dst, const void* src)
    {
        // clang-format off
        static const f32 values[16] = {
            0.0f, 0.0666667f, 0.133333f, 0.2f, 0.266667f, 0.333333f, 0.4f, 0.466667f,
            0.533333f, 0.6f, 0.666667f, 0.733333f, 0.8f, 0.866667f, 0.933333f, 1.0f,
        };
        // clang-format on
        f32* dstf = static_cast<f32*>(dst);
        const u8* srcu8 = static_cast<const u8*>(src);
        for(u64 i = 0; i < size; ++i) {
            u64 bit_index = i << 2;
            u64 index = bit_index >> 3;
            u64 remain = bit_index - (index << 3);
            u64 bit = (srcu8[index] >> remain) & 0x15UL;
            assert(remain < 8);
            if(5 <= remain) {
                switch(8 - remain) {
                case 1:
                    bit = bit | ((srcu8[index + 1] & 0x1UL) << 3UL);
                    break;
                case 2:
                    bit = bit | ((srcu8[index + 1] & 0x3UL) << 2UL);
                    break;
                case 3:
                    bit = bit | ((srcu8[index + 1] & 0x7UL) << 1UL);
                    break;
                }
            }
            assert(bit < 16);
            dstf[i] = values[bit];
        }
    }

    void copy5_f(u64 size, void* dst, const void* src)
    {
        // clang-format off
        static const f32 values[32] = {
            0.0f, 0.0322581f, 0.0645161f, 0.0967742f, 0.129032f, 0.16129f, 0.193548f, 0.225806f,
            0.258065f, 0.290323f, 0.322581f, 0.354839f, 0.387097f, 0.419355f, 0.451613f, 0.483871f,
            0.516129f, 0.548387f, 0.580645f, 0.612903f, 0.645161f, 0.677419f, 0.709677f, 0.741935f,
            0.774194f, 0.806452f, 0.83871f, 0.870968f, 0.903226f, 0.935484f, 0.967742f, 1.0f,
        };
        // clang-format on
        f32* dstf = static_cast<f32*>(dst);
        const u8* srcu8 = static_cast<const u8*>(src);
        for(u64 i = 0; i < size; ++i) {
            u64 bit_index = (i << 2) + i;
            u64 index = bit_index >> 3;
            u64 remain = bit_index - (index << 3);
            u64 bit = (srcu8[index] >> remain) & 0x31UL;
            assert(remain < 8);
            if(4 <= remain) {
                switch(8 - remain) {
                case 1:
                    bit = bit | ((srcu8[index + 1] & 0x1UL) << 4UL);
                    break;
                case 2:
                    bit = bit | ((srcu8[index + 1] & 0x3UL) << 3UL);
                    break;
                case 3:
                    bit = bit | ((srcu8[index + 1] & 0x7UL) << 2UL);
                    break;
                case 4:
                    bit = bit | ((srcu8[index + 1] & 0xFUL) << 1UL);
                    break;
                }
            }
            assert(bit < 32);
            dstf[i] = values[bit];
        }
    }

    void copy6_f(u64 size, void* dst, const void* src)
    {
        // clang-format off
        static const f32 values[64] = {
            0.0f, 0.015873f, 0.031746f, 0.047619f, 0.0634921f, 0.0793651f, 0.0952381f, 0.111111f,
            0.126984f, 0.142857f, 0.15873f, 0.174603f, 0.190476f, 0.206349f, 0.222222f, 0.238095f,
            0.253968f, 0.269841f, 0.285714f, 0.301587f, 0.31746f, 0.333333f, 0.349206f, 0.365079f,
            0.380952f, 0.396825f, 0.412698f, 0.428571f, 0.444444f, 0.460317f, 0.47619f, 0.492063f,
            0.507937f, 0.52381f, 0.539683f, 0.555556f, 0.571429f, 0.587302f, 0.603175f, 0.619048f,
            0.634921f, 0.650794f, 0.666667f, 0.68254f, 0.698413f, 0.714286f, 0.730159f, 0.746032f,
            0.761905f, 0.777778f, 0.793651f, 0.809524f, 0.825397f, 0.84127f, 0.857143f, 0.873016f,
            0.888889f, 0.904762f, 0.920635f, 0.936508f, 0.952381f, 0.968254f, 0.984127f, 1.0f,
        };
        // clang-format on
        f32* dstf = static_cast<f32*>(dst);
        const u8* srcu8 = static_cast<const u8*>(src);
        for(u64 i = 0; i < size; ++i) {
            u64 bit_index = i * 6;
            u64 index = bit_index >> 3;
            u64 remain = bit_index - (index << 3);
            u64 bit = (srcu8[index] >> remain) & 0x63UL;
            assert(remain < 8);
            if(3 <= remain) {
                switch(8 - remain) {
                case 1:
                    bit = bit | ((srcu8[index + 1] & 0x1UL) << 5UL);
                    break;
                case 2:
                    bit = bit | ((srcu8[index + 1] & 0x3UL) << 4UL);
                    break;
                case 3:
                    bit = bit | ((srcu8[index + 1] & 0x7UL) << 3UL);
                    break;
                case 4:
                    bit = bit | ((srcu8[index + 1] & 0xFUL) << 2UL);
                    break;
                case 5:
                    bit = bit | ((srcu8[index + 1] & 0x1FUL) << 1UL);
                    break;
                }
            }
            assert(bit < 64);
            dstf[i] = values[bit];
        }
    }

    void copy8_f(u64 size, void* dst, const void* src)
    {
        f32* dstf = static_cast<f32*>(dst);
        const u8* srcu8 = static_cast<const u8*>(src);
        for(u64 i = 0; i < size; ++i) {
            dstf[i] = srcu8[i] / 255.0f;
        }
    }

    void copyi8_f(u64 size, void* dst, const void* src)
    {
        f32* dstf = static_cast<f32*>(dst);
        const s8* srcs8 = static_cast<const s8*>(src);
        for(u64 i = 0; i < size; ++i) {
            dstf[i] = (0 <= srcs8[i]) ? srcs8[i] / 127.0f : srcs8[i] / 128.0f;
        }
    }

    void copyi16_f(u64 size, void* dst, const void* src)
    {
        f32* dstf = static_cast<f32*>(dst);
        const s16* srcs16 = static_cast<const s16*>(src);
        for(u64 i = 0; i < size; ++i) {
            dstf[i] = (0 <= srcs16[i]) ? srcs16[i] / 32767.0f : srcs16[i] / 32768.0f;
        }
    }

    void copyi32_f(u64 size, void* dst, const void* src)
    {
        f32* dstf = static_cast<f32*>(dst);
        const s32* srcs32 = static_cast<const s32*>(src);
        for(u64 i = 0; i < size; ++i) {
            f64 x = (0 <= srcs32[i]) ? srcs32[i] / 2'147'483'647.0 : srcs32[i] / 2'147'483'648.0;
            dstf[i] = static_cast<f32>(x);
        }
    }

    void copyi64_f(u64 size, void* dst, const void* src)
    {
        f32* dstf = static_cast<f32*>(dst);
        const s64* srcs64 = static_cast<const s64*>(src);
        for(u64 i = 0; i < size; ++i) {
            f64 x = (0 <= srcs64[i]) ? srcs64[i] / 9'223'372'036'854'775'807.0 : srcs64[i] / 9'223'372'036'854'775'808.0;
            dstf[i] = static_cast<f32>(x);
        }
    }

    void copyf16_f(u64 size, void* dst, const void* src)
    {
        f32* dstf = static_cast<f32*>(dst);
        const s16* srcs16 = static_cast<const s16*>(src);
        u64 qsize = (size >> 2) << 2;
        __declspec(align(16)) f32 result[4];
        for(u64 i = 0; i < qsize; i += 4) {
            __m128i x = _mm_loadl_epi64((__m128i*)&srcs16[i]);
            __m128 f = _mm_cvtph_ps(x);
            _mm_store_ps(result, f);
            dstf[i + 0] = result[0];
            dstf[i + 1] = result[1];
            dstf[i + 2] = result[2];
            dstf[i + 3] = result[3];
        }
        for(u64 i = qsize; i < size; ++i) {
            __m128i x = _mm_set1_epi16(srcs16[i]);
            __m128 f = _mm_cvtph_ps(x);
            _mm_store_ps(result, f);
            dstf[i] = result[0];
        }
    }

    void copyf32_f(u64 size, void* dst, const void* src)
    {
        ::memcpy(dst, src, sizeof(f32) * size);
    }

    void copyf64_f(u64 size, void* dst, const void* src)
    {
        f32* dstf = static_cast<f32*>(dst);
        const double* src64 = static_cast<const double*>(src);
        for(u64 i = 0; i < size; ++i) {
            dstf[i] = static_cast<f32>(src64[i]);
        }
    }
} // namespace util

void* allocate(size_t size, size_t align)
{
    return mi_malloc_aligned(size, align);
}

void deallocate(void* ptr, size_t align)
{
    mi_free_aligned(ptr, align);
}

u32 next_prime(u32 x)
{
    // clang-format off
    static constexpr u32 table[] = {
        5UL, 11UL, 17UL, 29UL, 37UL,
        53UL, 67UL, 79UL, 97UL, 131UL,
        193UL, 257UL, 389UL, 521UL, 769UL,
        1031UL, 1543UL, 2053UL, 3079UL, 6151UL,
        12289UL, 24593UL, 49157UL, 98317UL, 196613UL,
        393241UL, 786433UL, 1572869UL, 3145739UL, 6291469UL,
        12582917UL, 25165843UL, 50331653UL, 100663319UL, 201326611UL,
        402653189UL, 805306457UL, 1610612741UL, 3221225473UL, 4294967291UL,
    };
    static constexpr u32 size = sizeof(table)/sizeof(table[0]);
    // clang-format on

    const u32* const prime_list_begin = table;
    const u32* const prime_list_end = prime_list_begin + size;
    const u32* bound = lower_bound(prime_list_begin, prime_list_end, x);
    if(bound == prime_list_end) {
        --bound;
    }
    return *bound;
}

//--- Timer
//-----------------------------------------------------------
Timer::Timer(s64& duration)
    : duration_(duration)
    , start_(std::chrono::high_resolution_clock::now())
{
    duration_ = 0;
}

Timer::~Timer()
{
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::duration duration = end - start_;
    duration_ = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

//--- Memory
//-----------------------------------------------------------
Memory::Memory()
    : size_(0)
    , ptr_(nullptr)
{
}

Memory::Memory(Memory&& other)
    : size_(other.size_)
    , ptr_(other.ptr_)
{
    other.size_ = 0;
    other.ptr_ = nullptr;
}

Memory::Memory(u64 size)
    : size_(size)
    , ptr_(nullptr)
{
}

Memory::~Memory()
{
    deallocate(ptr_);
    ptr_ = nullptr;
}

Memory& Memory::operator=(Memory&& other)
{
    if(this != &other) {
        deallocate(ptr_);
        size_ = other.size_;
        ptr_ = other.ptr_;
        other.size_ = 0;
        other.ptr_ = nullptr;
    }
    return *this;
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
#    ifdef DNNL_WITH_SYCL
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
#    endif
#    if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if(eng.get_kind() == dnnl::engine::kind::gpu) {
            void* mapped_ptr = mem.map_data();
            if(nullptr != mapped_ptr) {
                ::memcpy(mapped_ptr, src, size);
            }
            mem.unmap_data(mapped_ptr);
            return;
        }
#    endif

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
#    ifdef DNNL_WITH_SYCL
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
#    endif
#    if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if(eng.get_kind() == dnnl::engine::kind::gpu) {
            void* mapped_ptr = mem.map_data();
            if(nullptr != mapped_ptr) {
                ::memset(mapped_ptr, x, size);
            }
            mem.unmap_data(mapped_ptr);
            return;
        }
#    endif

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
    , num_dims_(0)
    , bit_packed_(0)
    , dims_{}
    , data_(nullptr, CustomDeleter(true))
{
}

Tensor::Tensor(ggml_type type, std::initializer_list<u64> dimensions)
    : type_(type)
    , num_dims_(static_cast<u16>(dimensions.size()))
    , bit_packed_(0)
    , data_(nullptr, CustomDeleter(true))
{
    assert(0 < num_dims_ && num_dims_ <= GGML_MAX_DIMS);
    u64 total_size = 1;
    u32 i = 0;
    for(const u64& x: dimensions) {
        dims_[i] = x;
        total_size *= x;
        ++i;
    }
    u64 total_in_bytes = total_size * get_byte_size_aligned(type_);
    data_ = std::unique_ptr<u8[], CustomDeleter>(new u8[total_in_bytes], CustomDeleter(false));
}

Tensor::Tensor(ggml_type type, std::initializer_list<u64> dimensions, const void* data)
    : type_(type)
    , num_dims_(static_cast<u16>(dimensions.size()))
    , data_(static_cast<u8*>(const_cast<void*>(data)), CustomDeleter(true))
    , bit_packed_(1)
{
    assert(0 < num_dims_ && num_dims_ <= GGML_MAX_DIMS);
    assert(nullptr != data);
    u32 i = 0;
    for(const u64& x: dimensions) {
        dims_[i] = x;
        ++i;
    }
}

Tensor::Tensor(ggml_type type, const Tensor& shape)
    : type_(type)
    , num_dims_(shape.num_dims_)
    , bit_packed_(0)
    , data_(nullptr, CustomDeleter(true))
{
    type_ = type;
    bit_packed_ = 0;
    u64 total_size = 1;
    for(u32 i = 0; i < num_dims_; ++i) {
        dims_[i] = shape.dims_[i];
        total_size *= shape.dims_[i];
    }
    u64 total_in_bytes = total_size * get_byte_size_aligned(type_);
    data_ = std::unique_ptr<u8[], CustomDeleter>(new u8[total_in_bytes], CustomDeleter(false));
}

Tensor::Tensor(ggml_type type, const Tensor& shape, const void* data)
    : type_(type)
    , num_dims_(static_cast<u16>(shape.num_dims()))
    , data_(static_cast<u8*>(const_cast<void*>(data)), CustomDeleter(true))
    , bit_packed_(1)
{
    assert(0 < num_dims_ && num_dims_ <= GGML_MAX_DIMS);
    assert(nullptr != data);
    for(u32 i = 0; i < num_dims_; ++i) {
        dims_[i] = shape.dims_[i];
    }
}

Tensor::Tensor(Tensor&& other)
    : type_(other.type_)
    , num_dims_(other.num_dims_)
    , bit_packed_(other.bit_packed_)
    , data_(std::move(other.data_))
{
    for(u16 i = 0; i < num_dims_; ++i) {
        dims_[i] = other.dims_[i];
    }
    other.num_dims_ = 0;
}

Tensor::~Tensor()
{
}

Tensor& Tensor::operator=(Tensor&& other)
{
    if(this != &other) {
        type_ = other.type_;
        num_dims_ = other.num_dims_;
        bit_packed_ = other.bit_packed_;
        ::memcpy(dims_, other.dims_, sizeof(u64) * GGML_MAX_DIMS);
        data_ = std::move(other.data_);
        other.num_dims_ = 0;
    }
    return *this;
}

ggml_type Tensor::type() const
{
    return type_;
}

u32 Tensor::num_dims() const
{
    return num_dims_;
}

u64 Tensor::total_size() const
{
    u64 total_size = 1;
    for(u32 i = 0; i < num_dims_; ++i) {
        total_size *= dims_[i];
    }
    return total_size;
}

u64 Tensor::total_bytes() const
{
    u64 total = total_size();
    if(bit_packed_) {
        u64 total_bites = total * get_bit_size(type_);
        u64 total_bytes = (total_bites + 0x7UL) >> 3UL;
        return total_bytes;
    } else {
        return total * get_byte_size_aligned(type_);
    }
}

u64 Tensor::size(u32 index) const
{
    assert(index < num_dims_);
    return dims_[index];
}

void Tensor::resize(std::initializer_list<u64> dimensions) noexcept
{
    assert(1 <= dimensions.size() && dimensions.size() <= GGML_MAX_DIMS);
    u32 i = 0;
    u64 total = 1;
    for(const u64& x: dimensions) {
        total *= x;
        dims_[i] = x;
        ++i;
    }
    assert(total == total_size());
    num_dims_ = static_cast<u16>(dimensions.size());
}

bool is_same_shape(const Tensor& x0, const Tensor& x1)
{
    if(x0.num_dims() != x1.num_dims()) {
        return false;
    }
    for(u32 i = 0; i < x0.num_dims(); ++i) {
        if(x0.size(i) != x1.size(i)) {
            return false;
        }
    }
    return true;
}

namespace op
{
    Tensor convertF32(const Tensor& input)
    {
        if(ggml_type::GGML_TYPE_F32 == input.type()){
            Tensor result(ggml_type::GGML_TYPE_F32, input, input.data<void>());
            return result;
        }
        u64 size = input.total_size();
        Tensor result(ggml_type::GGML_TYPE_F32, input);
        switch(input.type()) {
        case ggml_type::GGML_TYPE_F32:
            util::copyf32_f(size, result.data<void>(), input.data<void>());
            break;
        case ggml_type::GGML_TYPE_F16:{
            util::copyf16_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_Q4_0:{
            util::copy4_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_Q4_1:{
            util::copy4_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_Q5_0:{
            util::copy5_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_Q5_1:{
            util::copy5_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_Q8_0:{
            util::copy8_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_Q8_1:{
            util::copy8_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_Q2_K:{
            util::copy2_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_Q3_K:{
            util::copy3_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_Q4_K:{
            util::copy4_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_Q5_K:{
            util::copy5_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_Q6_K:{
            util::copy6_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_Q8_K:{
            util::copy8_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_IQ2_XXS:{
            util::copy2_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_IQ2_XS:{
            util::copy2_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_IQ3_XXS:{
            util::copy3_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_IQ1_S:{
            util::copy1_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_IQ4_NL:{
            util::copy4_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_IQ3_S:{
            util::copy3_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_IQ2_S:{
            util::copy2_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_IQ4_XS:{
            util::copy4_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_I8:{
            util::copyi8_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_I16:{
            util::copyi16_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_I32:{
            util::copyi32_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_I64:{
            util::copyi64_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_F64:{
            util::copyf64_f(size, result.data<void>(), input.data<void>());
            }
            break;
        case ggml_type::GGML_TYPE_IQ1_M:{
            util::copy1_f(size, result.data<void>(), input.data<void>());
            }
            break;
        default:
            assert(false);
            break;
        }
        return result;
    }

    f32 dot_product(u64 size, const f32* x0, const f32* x1)
    {
        __m128 sum = _mm_setzero_ps();
        u64 qsize = (size >> 2) << 2;
        for(u64 i = 0; i < qsize; i += 4) {
            __m128 v0 = _mm_loadu_ps(&x0[i]);
            __m128 v1 = _mm_loadu_ps(&x1[i]);
            sum = _mm_fmadd_ps(v0, v1, sum);
        }
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        __declspec(align(16)) f32 r[4];
        _mm_store_ps(r, sum);
        f32 result = r[0];
        for(u64 i = qsize; i < size; ++i) {
            f32 v0 = x0[i];
            f32 v1 = x1[i];
            result += v0 * v1;
        }
        return result;
    }

    f32 kahan_sum(u64 size, const f32* src)
    {
        u64 qsize = (size >> 2) << 2;
        f32 sum = 0.0f;
        {
            __m128 s = _mm_setzero_ps();
            __m128 c = _mm_setzero_ps();
            for(u64 i = 0; i < qsize; i += 4) {
                __m128 x = _mm_loadu_ps(&src[i]);
                __m128 y = _mm_sub_ps(x, c);
                __m128 t = _mm_add_ps(s, y);
                c = _mm_sub_ps(_mm_sub_ps(t, s), y);
                s = t;
            }
            __declspec(align(16)) f32 r[4];
            _mm_store_ps(r, s);
            sum = r[0] + r[1] + r[2] + r[3];
        }

        {
            f32 c = 0.0f;
            for(u64 i = qsize; i < size; ++i) {
                f32 x = src[i];
                f32 y = x - c;
                f32 t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
        }
        return sum;
    }

    f32 kahan_sum_squared(u64 size, const f32* src, f32 mean)
    {
        u64 qsize = (size >> 2) << 2;
        __m128 qmean = _mm_set1_ps(mean);
        f32 sum = 0.0f;
        {
            __m128 s = _mm_setzero_ps();
            __m128 c = _mm_setzero_ps();
            for(u64 i = 0; i < qsize; i += 4) {
                __m128 x = _mm_loadu_ps(&src[i]);
                x = _mm_sub_ps(x, qmean);
                x = _mm_mul_ps(x, x);
                __m128 y = _mm_sub_ps(x, c);
                __m128 t = _mm_add_ps(s, y);
                c = _mm_sub_ps(_mm_sub_ps(t, s), y);
                s = t;
            }
            __declspec(align(16)) f32 r[4];
            _mm_store_ps(r, s);
            sum = r[0] + r[1] + r[2] + r[3];
        }
        {
            f32 c = 0.0f;
            for(u64 i = qsize; i < size; ++i) {
                f32 x = src[i];
                x = x - mean;
                x = x * x;
                f32 y = x - c;
                f32 t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
        }
        return sum;
    }

    void normalize_vec(u64 size, f32* dst, const f32* src, const f32* weight, const f32* bias)
    {
        // calcurate standard deviation
        f32 sum = kahan_sum(size, src);
        const f32 mean = sum / size;
        const f32 sum_squared = kahan_sum_squared(size, src, mean);
        const f32 variance = sum_squared / size;
        const f32 stddev = std::sqrt(variance);

        // normalize
        static constexpr f32 eps = 1e-06f;
        u64 qsize = (size >> 2) << 2;
        {
            __m128 qeps = _mm_set1_ps(eps);
            __m128 qmean = _mm_set1_ps(mean);
            __m128 qstddev = _mm_set1_ps(stddev);
            __declspec(align(16)) f32 r[4];
            for(u64 i = 0; i < qsize; i += 4) {
                __m128 x = _mm_loadu_ps(&src[i]);
                __m128 w = _mm_loadu_ps(&weight[i]);
                __m128 b = _mm_loadu_ps(&bias[i]);
                __m128 normalized = _mm_div_ps(_mm_sub_ps(x, qmean), _mm_add_ps(qstddev, qeps));
                normalized = _mm_add_ps(_mm_mul_ps(normalized, w), b);
                _mm_store_ps(r, normalized);
                dst[i + 0] = r[0];
                dst[i + 1] = r[1];
                dst[i + 2] = r[2];
                dst[i + 3] = r[3];
            }

            for(u64 i = qsize; i < size; ++i) {
                f32 x = src[i];
                f32 w = weight[i];
                f32 b = bias[i];

                // epsilon added to standard deviation for preventing division by zero.
                f32 normalized = ((x - mean) / (stddev + eps)) * w + b;
                dst[i] = normalized;
            }
        }
    }

    Tensor normalize(const Tensor& input, const Tensor& weight, const Tensor& bias)
    {
        assert(ggml_type::GGML_TYPE_F32 == input.type());

        Tensor result(ggml_type::GGML_TYPE_F32, {input.size(0), input.size(1)});

        Tensor weightf = std::move(convertF32(weight));
        Tensor biasf = std::move(convertF32(bias));
        u64 offset = 0;
        for(u64 i = 0; i < input.size(0); ++i) {
            const f32* src = input.data<f32>() + offset;
            const f32* w = weightf.data<f32>();
            const f32* b = biasf.data<f32>();
            f32* dst = result.data<f32>() + offset;

            normalize_vec(input.size(1), dst, src, w, b);
            offset += input.size(1);
        }
        return std::move(result);
    }

    Tensor embed_tokens(const Tensor& emb_weight, const Tensor& tokens)
    {
        assert(ggml_type::GGML_TYPE_I32 == tokens.type());
        const u64 d_embed = emb_weight.size(1);
        u64 total_size = tokens.total_size();
        Tensor result(ggml_type::GGML_TYPE_F32, {total_size, d_embed});
        for(u64 i = 0; i < total_size; ++i) {
            const s32 emb_index = tokens.data<s32>()[i];
            const u64 emb_offset = emb_index * d_embed;
            const u64 out_offset = i * d_embed;
            const f32* src = emb_weight.data<f32>() + emb_offset;
            f32* dst = result.data<f32>() + out_offset;
            ::memcpy(dst, src, sizeof(f32) * d_embed);
        }
        return result;
    }

    Tensor embed_projection(const Tensor& input, const Tensor& emb_weight)
    {
        const u64 n_ctx = input.size(0);
        const u64 n_vocab = emb_weight.size(0);
        const u64 n_embed = emb_weight.size(1);

        const u64 offset = (n_ctx - 1) * n_embed;
        Tensor result(ggml_type::GGML_TYPE_F32, {n_vocab});
        for(u64 i = 0; i < n_vocab; ++i) {
            const f32* emb = emb_weight.data<f32>() + i * n_embed;
            result.data<f32>()[i] = dot_product(n_embed, input.data<f32>() + offset, emb);
        }
        return result;
    }

    Tensor gelu(const Tensor& input)
    {
        assert(ggml_type::GGML_TYPE_F32 == input.type());
        const u64 n_vectors = input.size(0);

        Tensor result(ggml_type::GGML_TYPE_F32, {n_vectors});
        u64 end = input.total_size();
        for(u64 i = 0; i < end; ++i) {
            f32 x = input.data<f32>()[i];
            f32 res = 0.5f * x
                      * (1.0f + std::tanh(0.79788456079f // std::sqrt(2.0f / 3.141592653589793f)
                                          * (x + 0.044715f * std::pow(x, 3.0f))));
            result.data<f32>()[i] = res;
        }
        return result;
    }

    void vec_add(u64 size, f32* dst, const f32* src0, const f32* src1)
    {
        u64 qsize = (size >> 2) << 2;
        for(u64 i = 0; i < qsize; i += 4) {
            __m128 x0 = _mm_loadu_ps(&src0[i]);
            __m128 x1 = _mm_loadu_ps(&src1[i]);
            __m128 sum = _mm_add_ps(x0, x1);
            _mm_storeu_ps(&dst[i], sum);
        }
        for(u64 i = qsize; i < size; ++i) {
            f32 x0 = src0[i];
            f32 x1 = src1[i];
            dst[i] = x0 + x1;
        }
    }

    Tensor add(const Tensor& x0, const Tensor& x1)
    {
        assert(ggml_type::GGML_TYPE_F32 == x0.type());
        assert(ggml_type::GGML_TYPE_F32 == x1.type());
        const u64 nrows = x0.size(0);
        const u64 ncols = x0.size(1);

        Tensor result(ggml_type::GGML_TYPE_F32, {nrows, ncols});
        for(u64 i = 0; i < nrows; ++i) {
            u64 offset = i * ncols;
            const f32* row0 = x0.data<f32>() + offset;
            const f32* row1 = x1.data<f32>() + offset;
            f32* dst = result.data<f32>() + offset;
            vec_add(ncols, dst, row0, row1);
        }
        return result;
    }

    Tensor affine_proj_2d(const Tensor& input, const Tensor& weight, const Tensor& bias)
    {
        const u64 nrows0 = input.size(0);
        const u64 ncols = input.size(1);
        const u64 nrows1 = weight.size(0);

        Tensor result(ggml_type::GGML_TYPE_F32, {nrows0, nrows1});
        for(u64 r0 = 0; r0 < nrows0; ++r0) {
            for(u64 r1 = 0; r1 < nrows1; ++r1) {
                const f32* row0 = input.data<f32>() + r0 * ncols;
                const f32* row1 = weight.data<f32>() + r1 * ncols;
                f32 a = dot_product(ncols, row0, row1);
                f32 b = bias.data<f32>()[r1];
                result.data<f32>()[r0 * nrows1 + r1] = a + b;
            }
        }
        return result;
    }

    Tensor qk_masked_attn_matmul(
        const Tensor& q,
        const Tensor& k,
        const u64 n_heads)
    {
        const u64 q_nrows = q.size(0);
        const u64 ncols = q.size(1);
        const u64 k_nrows = k.size(0);
        const u64 d_head = ncols / n_heads;
        const f32 scale_factor = 1.0f / ::sqrtf(static_cast<f32>(d_head));

        Tensor result(ggml_type::GGML_TYPE_F32, {n_heads, q_nrows, q_nrows});
        for(u64 h = 0; h < n_heads; ++h) {
            for(u64 qrow = 0; qrow < q_nrows; ++qrow) {
                const u64 qrow_offset = h * d_head + qrow * ncols;
                // We only compute dot products that are not masked. 'k_max' represents
                // the number of dot products that we should compute for the current q row.
                const u64 kmax = qrow + 1;
                for(u64 kcol = 0; kcol < kmax; ++kcol) {
                    const u64 krow_offset = h * d_head + kcol * ncols;
                    const f32 dot = dot_product(d_head, q.data<f32>() + qrow_offset, k.data<f32>() + krow_offset);
                    u64 index = h * q_nrows * k_nrows + qrow * k_nrows + kcol;
                    result.data<f32>()[index] = dot * scale_factor;
                }
            }
        }

        // Do the masking.
        for(u64 h = 0; h < n_heads; ++h) {
            for(u64 qrow = 0; qrow < q_nrows; ++qrow) {
                const u64 kcol_start = qrow + 1;
                for(u64 kcol = kcol_start; kcol < k_nrows; ++kcol) {
                    const u64 index = h * q_nrows * k_nrows + qrow * k_nrows + kcol;
                    result.data<f32>()[index] = -std::numeric_limits<f32>::infinity();
                }
            }
        }
        return result;
    }

    void qk_softmax(Tensor& qk, u64 n_heads)
    {
        const u64 q_ctx = qk.size(1);
        const u64 k_ctx = qk.size(2);

        for(u64 h = 0; h < n_heads; ++h) {
            for(u64 qrow = 0; qrow < q_ctx; ++qrow) {
                f32 max_value = -std::numeric_limits<f32>::infinity();
                const u64 base = h * q_ctx * k_ctx + qrow * k_ctx;
                for(u64 i = 0; i < k_ctx; ++i) {
                    f32 x = qk.data<f32>()[base + i];
                    if(max_value < x) {
                        max_value = x;
                    }
                } // for(u64 i

                f32 sum_exp = 0.0f;
                for(u64 i = 0; i < k_ctx; ++i) {
                    f32 x = qk.data<f32>()[base + i];
                    f32 exp_value = ::expf(x - max_value);
                    qk.data<f32>()[base + i] = exp_value;
                    sum_exp += exp_value;
                } // for(u64 i
                f32 inv_sum_exp = 1.0f / sum_exp;
                for(u64 i = 0; i < k_ctx; ++i) {
                    f32 x = qk.data<f32>()[base + i];
                    qk.data<f32>()[base + i] = x * inv_sum_exp;
                } // for(u64 i

            } // for(u64 qrow
        } // for(u64 h
    }

    void qkv_attn_matmul(Tensor& qkv, const Tensor& qk, const Tensor& v, u64 n_heads)
    {
        const u64 qk_nrows = qk.size(1);
        const u64 qk_ncols = qk.size(2);
        const u64 d_embed = v.size(1);
        const u64 d_head = d_embed / n_heads;

        for(u64 h = 0; h < n_heads; ++h) {
            for(u64 qkrow = 0; qkrow < qk_nrows; ++qkrow) {
                for(u64 vcol = 0; vcol < d_head; ++vcol) {
                    f32 dot = 0.0f;
                    for(u64 i = 0; i < qk_ncols; ++i) {
                        u64 qk_i = h * qk_nrows * qk_ncols + qkrow * qk_ncols + i;
                        u64 v_i = h * d_head + i * d_embed + vcol;
                        f32 qkw = qk.data<f32>()[qk_i];
                        f32 vw = v.data<f32>()[v_i];
                        dot += qkw * vw;
                    }
                    u64 qkv_i = h * d_head + qkrow * d_embed + vcol;
                    qkv.data<f32>()[qkv_i] = dot;
                } // for(u64 vcol
            } // for(u64 qkrow
        } // for(u64 h
    }

} // namespace op

//--- LayerNorm
//-----------------------------------------------------------
LayerNorm::LayerNorm()
    :duration_(0)
{
}

LayerNorm::LayerNorm(
    ggml_type type,
    u32 d_in,
    u32 d_out,
    void* weight,
    void* bias)
    : duration_(0)
    , weight_(type, {d_out, d_in}, weight)
    , bias_(type, {d_out}, bias)
{
}

LayerNorm::~LayerNorm()
{
}

LayerNorm::LayerNorm(LayerNorm&& other)
    : duration_(0)
    , weight_(std::move(other.weight_))
    , bias_(std::move(other.bias_))
{
}

LayerNorm& LayerNorm::operator=(LayerNorm&& other)
{
    if(this != &other) {
        duration_ = 0;
        weight_ = std::move(other.weight_);
        bias_ = std::move(bias_);
    }
    return *this;
}

Tensor LayerNorm::forward(const Tensor& input)
{
    assert(weight_.size(0) == input.size(1));
    Timer timer(duration_);

    Tensor weight = op::convertF32(weight_);
    Tensor bias = op::convertF32(bias_);
    return op::normalize(input, weight, bias);
}

//--- Embedding
//-----------------------------------------------------------
Embedding::Embedding()
:duration_(0)
{
}

Embedding::Embedding(
    ggml_type type,
    u32 n_vocab,
    u32 d_embed,
    void* weight)
    : duration_(0)
    ,weight_(type, {n_vocab, d_embed}, weight)
{
}

Embedding::~Embedding()
{
}

Embedding::Embedding(Embedding&& other)
    : duration_(0)
    , weight_(std::move(other.weight_))
{
}

Embedding& Embedding::operator=(Embedding&& other)
{
    if(this != &other) {
        duration_ = 0;
        weight_ = std::move(other.weight_);
    }
    return *this;
}

Tensor Embedding::forward(const Tensor& input)
{
    assert(ggml_type::GGML_TYPE_I32 == input.type());
    assert(input.num_dims() == 1);
    Timer timer(duration_);
    Tensor weight = op::convertF32(weight_);
    return op::embed_tokens(input, weight);
}

Tensor Embedding::forward_proj(const Tensor& input)
{
    assert(input.num_dims() == 2);
    assert(weight_.size(1) == input.size(1));
    Timer timer(duration_);
    Tensor weight = op::convertF32(weight_);
    return op::embed_projection(input, weight);
}

//--- PositionalEmbedding
//-----------------------------------------------------------
PositionalEmbedding::PositionalEmbedding()
    :duration_(0)
{
}

PositionalEmbedding::PositionalEmbedding(
    ggml_type type,
    u64 max_context,
    u64 d_embed,
    void* weight)
    : duration_(0)
    ,weight_(type, {max_context, d_embed}, weight)
{
}

PositionalEmbedding::~PositionalEmbedding()
{
}

PositionalEmbedding::PositionalEmbedding(PositionalEmbedding&& other)
    : duration_(0)
    , weight_(std::move(other.weight_))
{
}

PositionalEmbedding& PositionalEmbedding::operator=(PositionalEmbedding&& other)
{
    if(this != &other) {
        duration_ = 0;
        weight_ = std::move(other.weight_);
    }
    return *this;
}

Tensor PositionalEmbedding::forward(u64 num_context)
{
    assert(num_context <= weight_.size(0));
    Timer timer(duration_);
    Tensor weight = op::convertF32(weight_);
    Tensor result(ggml_type::GGML_TYPE_F32, {num_context, weight_.size(1)});
    for(u64 i = 0; i < num_context; ++i) {
        u64 offset = i * weight_.size(1);
        const f32* src = weight.data<f32>() + offset;
        f32* dst = result.data<f32>() + offset;
        ::memcpy(dst, src, sizeof(f32) * weight_.size(1));
    }
    return result;
}

//--- GELU
//-----------------------------------------------------------
GELU::GELU()
    :duration_(0)
{
}

GELU::~GELU()
{
}

GELU::GELU(GELU&& /*other*/)
    : duration_(0)
{
}

GELU& GELU::operator=(GELU&& /*other*/)
{
    duration_ = 0;
    return *this;
}

Tensor GELU::forward(const Tensor& input)
{
    Timer timer(duration_);
    return op::gelu(input);
}

//--- Residual
//-----------------------------------------------------------
Residual::Residual()
    :duration_(0)
{
}

Residual::~Residual()
{
}

Residual::Residual(Residual&& /*other*/)
    : duration_(0)
{
}

Residual& Residual::operator=(Residual&& /*other*/)
{
    duration_ = 0;
    return *this;
}

Tensor Residual::forward(const Tensor& input0, const Tensor& input1)
{
    assert(input0.type() == input1.type());
    assert(input0.num_dims() == 2);
    assert(input1.num_dims() == 2);
    assert(is_same_shape(input0, input1));
    Timer timer(duration_);
    return op::add(input0, input1);
}

//--- Linear
//-----------------------------------------------------------
Linear::Linear()
    :duration_(0)
{
}

Linear::Linear(ggml_type type, u64 d_in, u64 d_out, void* weight, void* bias)
    : duration_(0)
    , weight_(type, {d_out, d_in}, weight)
    , bias_(type, {d_out}, bias)
{
}

Linear::~Linear()
{
}

Linear::Linear(Linear&& other)
    : duration_(0)
    , weight_(std::move(other.weight_))
    , bias_(std::move(other.bias_))
{
}

Linear& Linear::operator=(Linear&& other)
{
    if(this != &other) {
        duration_ = 0;
        weight_ = std::move(other.weight_);
        bias_ = std::move(other.bias_);
    }
    return *this;
}

Tensor Linear::forward(const Tensor& input)
{
    assert(2 == input.num_dims());
    assert(input.size(1) == weight_.size(1));
    Timer timer(duration_);
    Tensor weight = op::convertF32(weight_);
    Tensor bias = op::convertF32(bias_);
    return op::affine_proj_2d(input, weight, bias);
}

//--- MultiHeadSelfAttn
//-----------------------------------------------------------
MultiHeadSelfAttn::MultiHeadSelfAttn()
    :duration_(0)
    ,n_heads_(0)
{
}

MultiHeadSelfAttn::MultiHeadSelfAttn(
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
    void* qkv_proj_b)
    : duration_(0)
    ,n_heads_(n_heads)
    , query_(type, n_embed, n_embed, query_w, query_b)
    , key_(type, n_embed, n_embed, key_w, key_b)
    , value_(type, n_embed, n_embed, value_w, value_b)
    , qkv_proj_(type, n_embed, n_embed, qkv_proj_w, qkv_proj_b)
{
}

MultiHeadSelfAttn::~MultiHeadSelfAttn()
{
}

MultiHeadSelfAttn::MultiHeadSelfAttn(MultiHeadSelfAttn&& other)
    : duration_(0)
    , n_heads_(other.n_heads_)
    , query_(std::move(other.query_))
    , key_(std::move(other.key_))
    , value_(std::move(other.value_))
    , qkv_proj_(std::move(other.qkv_proj_))
{
}

MultiHeadSelfAttn& MultiHeadSelfAttn::operator=(MultiHeadSelfAttn&& other)
{
    if(this != &other){
    duration_ = 0;
    n_heads_ = other.n_heads_;
    query_ = std::move(other.query_);
    key_ = std::move(other.key_);
    value_ = std::move(other.value_);
    qkv_proj_ = std::move(other.qkv_proj_);

    other.n_heads_ = 0;
    }
    return *this;
}

Tensor MultiHeadSelfAttn::forward(const Tensor& input)
{
    assert(2 == input.num_dims());
    Timer timer(duration_);

    Tensor q = query_.forward(input);
    Tensor k = key_.forward(input);
    Tensor v = value_.forward(input);

    Tensor qkv = masked_qkv_attn(q, k, v);
    Tensor out = qkv_proj_.forward(qkv);
    return out;
}

Tensor MultiHeadSelfAttn::masked_qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v)
{
    const u64 n_ctx = q.size(0);
    const u64 d_embed = q.size(1);
    Tensor qkv(ggml_type::GGML_TYPE_F32, {n_ctx, d_embed});

    Tensor qk = op::qk_masked_attn_matmul(q, k, n_heads_);
    op::qk_softmax(qk, n_heads_);
    op::qkv_attn_matmul(qkv, v, qk, n_heads_);

    return qkv;
}

//--- ResidualAttnBlock
//-----------------------------------------------------------
ResidualAttnBlock::ResidualAttnBlock()
    :duration_(0)
{
}

//ResidualAttnBlock::ResidualAttnBlock(ggml_type type, u64 n_attn_heads, u64 d_embed, u64 d_mlp, u64 max_ctx)
//    : attn_ln_(type, max_ctx, d_embed,LayerNorm(max_ctx, d_embed)},
//      attn{MultiHeadSelfAttn(n_attn_heads, d_embed, max_ctx)},
//      inp_res{Residual(max_ctx, d_embed)},
//      mlp_ln{LayerNorm(max_ctx, d_embed)},
//      mlp_fc{Linear(d_embed, d_mlp, max_ctx)},
//      gelu{GELU(max_ctx, d_mlp, /*cache_ctx_acv=*/true)},
//      mlp_proj{Linear(d_mlp, d_embed, max_ctx)},
//      attn_res{Residual(max_ctx, d_embed)}
//{
//}

ResidualAttnBlock::~ResidualAttnBlock()
{
}

ResidualAttnBlock::ResidualAttnBlock(ResidualAttnBlock&& other)
    : duration_(0)
    , attn_ln_(std::move(other.attn_ln_))
    , attn_(std::move(other.attn_))
    , inp_res_(std::move(other.inp_res_))
    , mlp_ln_(std::move(other.mlp_ln_))
    , mlp_fc_(std::move(other.mlp_fc_))
    , gelu_(std::move(other.gelu_))
    , mlp_proj_(std::move(other.mlp_proj_))
    , attn_res_(std::move(other.attn_res_))
{
}

ResidualAttnBlock& ResidualAttnBlock::operator=(ResidualAttnBlock&& other)
{
    if(this != &other) {
        duration_ = 0;
        attn_ln_ = std::move(other.attn_ln_);
        attn_ = std::move(other.attn_);
        inp_res_ = std::move(other.inp_res_);
        mlp_ln_ = std::move(other.mlp_ln_);
        mlp_fc_ = std::move(other.mlp_fc_);
        gelu_ = std::move(other.gelu_);
        mlp_proj_ = std::move(other.mlp_proj_);
        attn_res_ = std::move(other.attn_res_);
    }
    return *this;
}

Tensor ResidualAttnBlock::forward(const Tensor& input)
{
    Timer timer(duration_);
    Tensor attn_ln_out = attn_ln_.forward(input);
    Tensor attn_out = attn_.forward(attn_ln_out);
    Tensor inp_res_out = inp_res_.forward(input, attn_out);
    Tensor mlp_ln_out = mlp_ln_.forward(inp_res_out);
    Tensor mlp_fc_out = mlp_fc_.forward(mlp_ln_out);
    Tensor gelu_out = gelu_.forward(mlp_fc_out);
    Tensor mlp_proj_out = mlp_proj_.forward(gelu_out);
    Tensor result = attn_res_.forward(inp_res_out, mlp_proj_out);
    return result;
}

//--- RMSNorm
//-----------------------------------------------------------
RMSNorm::RMSNorm(ggml_type type, u32 dimensions, f32 epsilon)
    : duration_(0)
    ,epsilon_(epsilon)
    , weight_(type, {dimensions})
{
}

RMSNorm::~RMSNorm()
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
{
}

Llama2::~Llama2()
{
}

Llama2& Llama2::operator=(Llama2&& other)
{
    if(this == &other) {
        return *this;
    }
    config_ = other.config_;
    return *this;
}
} // namespace cppgpt

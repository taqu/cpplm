#include "cppgpt.h"
#include <cmath>
#include <functional>
#include <algorithm>
#include <bit>
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

// void* operator new(std::size_t /*size*/, void* ptr) noexcept
//{
//     return ptr;
// }

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

// void operator delete(void* ptr, void*) noexcept
//{
//     (void)ptr;
// }

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

// void* operator new[](std::size_t /*size*/, void* ptr) noexcept
//{
//     return ptr;
// }

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

// void operator delete[](void* ptr, void*) noexcept
//{
//     (void)ptr;
// }

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

void* allocate(size_t size, size_t align)
{
    return mi_malloc_aligned(size, align);
}

void deallocate(void* ptr, size_t align)
{
    mi_free_aligned(ptr, align);
}

namespace
{
    //--- SplitMix
//--------------------------------------------
uint64_t SplitMix_next(uint64_t& state)
{
    state += 0x9E3779B97f4A7C15ULL;
    uint64_t t = state;
    t = (t ^ (t >> 30)) * 0xBF58476D1CE4E5B9ULL;
    t = (t ^ (t >> 27)) * 0x94D049BB133111EBULL;
    return t ^ (t >> 31);
}

/**
     * @brief 32 bit right rotation
     * @param [in] x ... input
     * @param [in] r ... count of rotation
     * @return rotated
     */
    inline uint32_t rotr32(uint32_t x, uint32_t r)
    {
        return (x >> r) | (x << ((~r + 1) & 31U));
    }
}

//--- Random
//------------------------------------------------------------
Random::Random()
    : state_{DEFAULT_SEED64}
{
}

Random::Random(uint64_t seed)
{
    srand(seed);
}

Random::~Random()
{
}

void Random::srand(uint64_t seed)
{
    state_ = SplitMix_next(seed);
    while(0 == state_) {
        state_ = SplitMix_next(state_);
    }
}

uint32_t Random::rand()
{
    uint64_t x = state_;
    uint32_t c = static_cast<uint32_t>(x >> 59);
    state_ = x * Multiplier + Increment;
    x ^= x >> 18;
    return rotr32(static_cast<uint32_t>(x >> 27), c);
}

float Random::frand()
{
    constexpr int32_t lowExp = 0;
    constexpr int32_t highExp = 127;
    const uint32_t u = rand();
    const uint32_t b = u & 0xFFU;
    int32_t exponent = highExp - 1;
    if(0 == b) {
        exponent -= 8;
        while(true) {
            const uint32_t bits = rand();
            if(0 == bits) {
                exponent -= 32;
                if(exponent < lowExp) {
                    exponent = lowExp;
                    break;
                }
            } else {
                int32_t c = std::countr_zero(bits);
                exponent -= c;
                break;
            }
        }
    }else{
        int32_t c = std::countr_zero(b);
        exponent -= c;
    }
    const uint32_t mantissa = (u>>8)&0x7FFFFFUL;
    if(0==mantissa && (u>>31)){
        ++exponent;
    }
    return std::bit_cast<float,uint32_t>((exponent<<23)|mantissa);
}

//--- wyhash
//-----------------------------------------------------------
namespace
{
    /* The Unlicense
    This is free and unencumbered software released into the public domain.
    Anyone is free to copy, modify, publish, use, compile, sell, or
    distribute this software, either in source code form or as a compiled
    binary, for any purpose, commercial or non-commercial, and by any
    means.
    In jurisdictions that recognize copyright laws, the author or authors
    of this software dedicate any and all copyright interest in the
    software to the public domain. We make this dedication for the benefit
    of the public at large and to the detriment of our heirs and
    successors. We intend this dedication to be an overt act of
    relinquishment in perpetuity of all present and future rights to this
    software under copyright law.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
    OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
    OTHER DEALINGS IN THE SOFTWARE.
    For more information, please refer to <http://unlicense.org/>
    */
    // This is free and unencumbered software released into the public domain under The Unlicense (http://unlicense.org/)
    // main repo: https://github.com/wangyi-fudan/wyhash
    // author: 王一 Wang Yi <godspeed_china@yeah.net>
    // contributors: Reini Urban, Dietrich Epp, Joshua Haberman, Tommy Ettinger, Daniel Lemire, Otmar Ertl, cocowalla, leo-yuriev, Diego Barrios Romero, paulie-g, dumblob, Yann Collet, ivte-ms, hyb, James Z.M. Gao, easyaspi314 (Devin), TheOneric
    // endian macros

#ifndef WYHASH_LITTLE_ENDIAN
#    if defined(_WIN32) || defined(__LITTLE_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#        define WYHASH_LITTLE_ENDIAN 1
#    elif defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#        define WYHASH_LITTLE_ENDIAN 0
#    else
#        warning could not determine endianness! Falling back to little endian.
#        define WYHASH_LITTLE_ENDIAN 1
#    endif
#endif

#if defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__clang__)
#    define wyhash_likely(x) __builtin_expect((x), 1)
#    define wyhash_unlikely(x) __builtin_expect((x), 0)
#else
#    define wyhash_likely(x) (x)
#    define wyhash_unlikely(x) (x)
#endif

#if defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__clang__)
#    define wyhash_restrict __restrict__
#elif defined(_MSC_VER)
#    define wyhash_restrict __restrict
#else
#    define wyhash_restrict
#endif

// read functions
#if(WYHASH_LITTLE_ENDIAN)
    static inline u64 wyhash_read8(const u8* p)
    {
        u64 v;
        ::memcpy(&v, p, 8);
        return v;
    }

    static inline u64 wyhash_read4(const u8* p)
    {
        u32 v;
        ::memcpy(&v, p, 4);
        return v;
    }

#elif defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__clang__)
    static inline u64 wyhash_read8(const u8* p)
    {
        u64 v;
        ::memcpy(&v, p, 8);
        return __builtin_bswap64(v);
    }
    static inline u64 wyhash_read4(const u8* p)
    {
        u32 v;
        ::memcpy(&v, p, 4);
        return __builtin_bswap32(v);
    }
#elif defined(_MSC_VER)
    static inline u64 wyhash_read8(const u8* p)
    {
        u64 v;
        ::memcpy(&v, p, 8);
        return _byteswap_uint64(v);
    }
    static inline u64 wyhash_read4(const u8* p)
    {
        u32 v;
        ::memcpy(&v, p, 4);
        return _byteswap_ulong(v);
    }
#else
    static inline u64 wyhash_read8(const u8* p)
    {
        u64 v;
        ::memcpy(&v, p, 8);
        return (((v >> 56) & 0xff) | ((v >> 40) & 0xff00) | ((v >> 24) & 0xff0000) | ((v >> 8) & 0xff000000) | ((v << 8) & 0xff00000000) | ((v << 24) & 0xff0000000000) | ((v << 40) & 0xff000000000000) | ((v << 56) & 0xff00000000000000));
    }
    static inline u64 wyhash_read4(const u8* p)
    {
        u32 v;
        ::memcpy(&v, p, 4);
        return (((v >> 24) & 0xff) | ((v >> 8) & 0xff00) | ((v << 8) & 0xff0000) | ((v << 24) & 0xff000000));
    }
#endif

    static inline u64 wyhash_read3(const u8* p, size_t size)
    {
        return (((u64)p[0]) << 16) | (((u64)p[size >> 1]) << 8) | p[size - 1];
    }

    // 128bit multiply function
    static inline u64 wyhash_rot(u64 x)
    {
        return (x >> 32) | (x << 32);
    }
    static inline void wyhash_mum(u64* wyhash_restrict A, u64* wyhash_restrict B)
    {
#if(1)
        u64 hh = (*A >> 32) * (*B >> 32);
        u64 hl = (*A >> 32) * (u32)*B;
        u64 lh = (u32)*A * (*B >> 32);
        u64 ll = (u64)(u32)*A * (u32)*B;

        *A = wyhash_rot(hl) ^ hh;
        *B = wyhash_rot(lh) ^ ll;

#elif defined(__SIZEOF_INT128__)
        __uint128_t r = *A;
        r *= *B;
        *A = (u64)r;
        *B = (u64)(r >> 64U);

#elif defined(_MSC_VER) && defined(_M_X64)
        *A = _umul128(*A, *B, B);
#else
        u64 ha = *A >> 32;
        u64 hb = *B >> 32;
        u64 la = (u32)*A;
        u64 lb = (u32)*B;
        u64 hi, lo;
        u64 rh = ha * hb;
        u64 rm0 = ha * lb;
        u64 rm1 = hb * la;
        u64 rl = la * lb;
        u64 t = rl + (rm0 << 32);
        u64 c = t < rl;
        lo = t + (rm1 << 32);
        c += lo < t;
        hi = rh + (rm0 >> 32) + (rm1 >> 32) + c;
        *A = lo;
        *B = hi;
#endif
    }

    // multiply and xor mix function, aka MUM
    static inline u64 wyhash_mix(u64 A, u64 B)
    {
        wyhash_mum(&A, &B);
        return A ^ B;
    }

    // the default secret parameters
    static const u64 wyhash_param[4] = {0xA0761D6478BD642FULL, 0xE7037ED1A0B428DBULL, 0x8EBC6AF09C88C6E3ULL, 0x589965CC75374CC3ULL};

    static inline u32 wyhash_combine(u32 x0, u32 x1)
    {
        u64 t = x0 ^ 0x53C5CA59U;
        t *= x1 ^ 0x74743C1BU;
        x0 = static_cast<u32>(t);
        x1 = static_cast<u32>(t >> 32U);
        return x0 ^ x1;
    }

} // namespace

u32 wyhash32(size_t size, const void* key, u64 seed)
{
    seed = wyhash64(size, key, seed);
    return (u32)(seed - (seed >> 32U));
}

u64 wyhash64(size_t size, const void* key, u64 seed)
{
    const u64* secret = wyhash_param;
    const u8* p = (const u8*)key;
    seed ^= *secret;
    u64 a, b;
    if(wyhash_likely(size <= 16)) {
        if(wyhash_likely(4 <= size)) {
            a = (wyhash_read4(p) << 32) | wyhash_read4(p + ((size >> 3) << 2));
            b = (wyhash_read4(p + size - 4) << 32) | wyhash_read4(p + size - 4 - ((size >> 3) << 2));
        } else if(wyhash_likely(0 < size)) {
            a = wyhash_read3(p, size);
            b = 0;
        } else {
            a = b = 0;
        }
    } else {
        size_t i = size;
        if(wyhash_unlikely(48 < i)) {
            u64 see1 = seed;
            u64 see2 = seed;
            do {
                seed = wyhash_mix(wyhash_read8(p) ^ secret[1], wyhash_read8(p + 8) ^ seed);
                see1 = wyhash_mix(wyhash_read8(p + 16) ^ secret[2], wyhash_read8(p + 24) ^ see1);
                see2 = wyhash_mix(wyhash_read8(p + 32) ^ secret[3], wyhash_read8(p + 40) ^ see2);
                p += 48;
                i -= 48;
            } while(wyhash_likely(i > 48));
            seed ^= see1 ^ see2;
        }
        while(wyhash_unlikely(i > 16)) {
            seed = wyhash_mix(wyhash_read8(p) ^ secret[1], wyhash_read8(p + 8) ^ seed);
            i -= 16;
            p += 16;
        }
        a = wyhash_read8(p + i - 16);
        b = wyhash_read8(p + i - 8);
    }
    return wyhash_mix(secret[1] ^ size, wyhash_mix(a ^ secret[1], b ^ seed));
}

//--- prime numbers
//-----------------------------------------------------------
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
        if(ggml_type::GGML_TYPE_F32 == input.type()) {
            Tensor result(ggml_type::GGML_TYPE_F32, input, input.data<void>());
            return result;
        }
        u64 size = input.total_size();
        Tensor result(ggml_type::GGML_TYPE_F32, input);
        switch(input.type()) {
        case ggml_type::GGML_TYPE_F32:
            util::copyf32_f(size, result.data<void>(), input.data<void>());
            break;
        case ggml_type::GGML_TYPE_F16: {
            util::copyf16_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_Q4_0: {
            util::copy4_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_Q4_1: {
            util::copy4_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_Q5_0: {
            util::copy5_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_Q5_1: {
            util::copy5_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_Q8_0: {
            util::copy8_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_Q8_1: {
            util::copy8_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_Q2_K: {
            util::copy2_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_Q3_K: {
            util::copy3_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_Q4_K: {
            util::copy4_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_Q5_K: {
            util::copy5_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_Q6_K: {
            util::copy6_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_Q8_K: {
            util::copy8_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_IQ2_XXS: {
            util::copy2_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_IQ2_XS: {
            util::copy2_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_IQ3_XXS: {
            util::copy3_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_IQ1_S: {
            util::copy1_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_IQ4_NL: {
            util::copy4_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_IQ3_S: {
            util::copy3_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_IQ2_S: {
            util::copy2_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_IQ4_XS: {
            util::copy4_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_I8: {
            util::copyi8_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_I16: {
            util::copyi16_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_I32: {
            util::copyi32_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_I64: {
            util::copyi64_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_F64: {
            util::copyf64_f(size, result.data<void>(), input.data<void>());
        } break;
        case ggml_type::GGML_TYPE_IQ1_M: {
            util::copy1_f(size, result.data<void>(), input.data<void>());
        } break;
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

    Tensor affine_proj_2d(const Tensor& input, const Tensor& weight)
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
                result.data<f32>()[r0 * nrows1 + r1] = a;
            }
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
        }     // for(u64 h
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
            }     // for(u64 qkrow
        }         // for(u64 h
    }

    void matmul(f32* dst, const f32* x, const f32* w, u64 n, u64 d)
    {
        for(u64 i = 0; i < d; ++i) {
            f32 value = 0.0f;
            for(u64 j = 0; j < n; ++j) {
                value += w[i * n + j] * x[j];
            }
            dst[i] = value;
        }
    }

    void rmsnorm(u64 size, f32* dst, const f32* x, const f32* w, f32 epsilon)
    {
        // calculate sum of squares
        f32 ss = 0.0f;
        for(u64 i = 0; i < size; ++i) {
            ss += x[i] * x[i];
        }
        ss /= size;
        ss += epsilon;
        ss = 1.0f / ::sqrtf(ss);
        // normalize and scale
        for(u64 i = 0; i < size; ++i) {
            dst[i] = w[i] * (ss * x[i]);
        }
    }

    void softmax(u64 size, f32* x)
    {
        assert(0 < size);
        // find max value (for numerical stability)
        f32 max_value = x[0];
        for(u64 i = 1; i < size; ++i) {
            if(max_value < x[i]) {
                max_value = x[i];
            }
        }
        // exp and sum
        f32 sum = 0.0f;
        for(u64 i = 0; i < size; ++i) {
            x[i] = ::expf(x[i] - max_value);
            sum += x[i];
        }
        // normalize
        f32 inv_sum = 1.0f / sum;
        for(u64 i = 0; i < size; ++i) {
            x[i] *= inv_sum;
        }
    }

} // namespace op

//--- Embedding
//-----------------------------------------------------------
Embedding::Embedding()
    : duration_(0)
{
}

Embedding::Embedding(
    ggml_type type,
    u32 n_vocab,
    u32 d_embed,
    void* weight)
    : duration_(0)
    , weight_(type, {n_vocab, d_embed}, weight)
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
    : duration_(0)
{
}

PositionalEmbedding::PositionalEmbedding(
    ggml_type type,
    u64 max_context,
    u64 d_embed,
    void* weight)
    : duration_(0)
    , weight_(type, {max_context, d_embed}, weight)
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

//--- Residual
//-----------------------------------------------------------
Residual::Residual()
    : duration_(0)
{
}

Residual::~Residual()
{
}

void Residual::forward(Tensor& dst, const Tensor& src0, const Tensor& src1)
{
    Timer timer(duration_);
    f32* d = dst.data<f32>();
    const f32* s0 = src0.data<f32>();
    const f32* s1 = src1.data<f32>();
    for(u64 i = 0; i < src0.size(0); ++i) {
        d[i] = s0[i] + s1[i];
    }
}

//--- RMSNorm
//-----------------------------------------------------------
RMSNorm::RMSNorm()
    : duration_(0)
    , epsilon_(0.0f)
{
}

RMSNorm::RMSNorm(Tensor&& weight, f32 epsilon)
    : duration_(0)
    , epsilon_(epsilon)
    , weight_(std::move(weight))
{
}

RMSNorm::~RMSNorm()
{
}

void RMSNorm::forward(Tensor& dst, const Tensor& src)
{
    Timer timer(duration_);
    Tensor weight = op::convertF32(weight_);
    op::rmsnorm(weight.size(0), dst.data<f32>(), src.data<f32>(), weight.data<f32>(), epsilon_);
}

RMSNorm::RMSNorm(RMSNorm&& other)
    : duration_(0)
    , epsilon_(other.epsilon_)
    , weight_(std::move(other.weight_))
{
}

RMSNorm& RMSNorm::operator=(RMSNorm&& other)
{
    if(this != &other) {
        duration_ = 0;
        epsilon_ = other.epsilon_;
        weight_ = std::move(other.weight_);
    }
    return *this;
}

//--- SelfAttention
//-----------------------------------------------------------
SelfAttention::SelfAttention()
    : duration_(0)
{
}

SelfAttention::SelfAttention(
    Tensor&& query,
    Tensor&& key,
    Tensor&& value,
    Tensor&& qkv)
    : duration_(0)
    , query_(std::move(query))
    , key_(std::move(key))
    , value_(std::move(value))
    , qkv_proj_(std::move(qkv))
{
}

SelfAttention::~SelfAttention()
{
}

SelfAttention::SelfAttention(SelfAttention&& other)
    : duration_(0)
    , query_(std::move(other.query_))
    , key_(std::move(other.key_))
    , value_(std::move(other.value_))
    , qkv_proj_(std::move(other.qkv_proj_))

{
}

SelfAttention& SelfAttention::operator=(SelfAttention&& other)
{
    if(this != &other) {
        duration_ = 0;
        query_ = std::move(other.query_);
        key_ = std::move(other.key_);
        value_ = std::move(other.value_);
        qkv_proj_ = std::move(other.qkv_proj_);
    }
    return *this;
}

void SelfAttention::forward(
    const Config& config,
    u64 position,
    u64 layer_offset,
    Tensor& output,
    Tensor& input,
    Tensor& query,
    Tensor& key_cache,
    Tensor& value_cache,
    Tensor& attention)
{
    u64 dim = input.size(0);
    u64 kv_dim = key_.size(1);
    u64 n_heads = config.num_heads_;
    u64 kv_mul = config.num_heads_ / config.num_kv_heads_;
    u64 head_size = config.dimension_ / n_heads;
    u64 cache_offset = layer_offset + position * kv_dim;
    f32* q = query.data<f32>();
    f32* k = key_cache.data<f32>() + cache_offset;
    f32* v = value_cache.data<f32>() + cache_offset;

    // qkv matmuls for the current position
    {
        Tensor wq = op::convertF32(query_);
        op::matmul(q, input.data<f32>(), wq.data<f32>(), dim, dim);
    }
    {
        Tensor wk = op::convertF32(key_);
        op::matmul(k, input.data<f32>(), wk.data<f32>(), dim, kv_dim);
    }
    {
        Tensor wv = op::convertF32(key_);
        op::matmul(v, input.data<f32>(), wv.data<f32>(), dim, kv_dim);
    }
    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    {
        for(u64 i = 0; i < dim; i += 2) {
            u64 head_dim = i % head_size;
            f32 freq = 1.0f / ::powf(10000.0f, head_dim / (f32)head_size);
            f32 value = position * freq;
            f32 fcr = ::cosf(value);
            f32 fci = ::sinf(value);
            u32 rotn = i < kv_dim ? 2 : 1;
            for(u32 j = 0; j < rotn; ++j) {
                f32* vec = (0 == j) ? q : k;
                f32 v0 = vec[i + 0];
                f32 v1 = vec[i + 1];
                vec[i + 0] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
    }
    // multihead attention. iterate over all heads
    {
        ::memset(input.data<f32>(), 0, n_heads*head_size*sizeof(f32));
        f32 inv_head_size = 1.0f / ::sqrtf(static_cast<float>(head_size));
        for(u64 h = 0; h < n_heads; ++h) {
            const f32* tq = q + h * head_size;                               // query vector for this head
            f32* attn = attention.data<f32>() + h * config.sequence_length_; // attention scores for this head
            // iterate over all timesteps, including the current step
            for(u64 t = 0; t <= position; ++t) {
                f32* tk = k + layer_offset + t * kv_dim + (h / kv_mul) * head_size; // key vector for this head and at this timestep
                // calcurate the attention score as the dot product of q and k
                f32 score = 0.0f;
                for(u64 i = 0; i < head_size; ++i) {
                    score += q[i] * tk[i];
                }
                score *= inv_head_size;
                attn[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            op::softmax(position + 1, attn);

            // weighted sum of the values, store back to the current tensor
            f32* xb = input.data<f32>() + h * head_size;
            for(u64 t = 0; t <= position; ++t) {
                f32* value = value_cache.data<f32>() + layer_offset + t * kv_dim + (h / kv_mul) * head_size;
                f32 a = attn[t];
                // accumulate the weighted value
                for(u64 i = 0; i < head_size; ++i) {
                    xb[i] += a * value[i];
                }
            }
        } // for(u64 h
    }

    // final matmul to get the output of the attention
    {
        Tensor wo = op::convertF32(qkv_proj_);
        op::matmul(output.data<f32>(), input.data<f32>(), wo.data<f32>(), dim, dim);
    }
}

//--- FeedForwardSwiGLU
//-----------------------------------------------------------
FeedForwardSwiGLU::FeedForwardSwiGLU()
    : duration_(0)
{
}

FeedForwardSwiGLU::FeedForwardSwiGLU(
    Tensor&& ffn_down,
    Tensor&& ffn_gate,
    Tensor&& ffn_up,
    Tensor&& ffn_norm)
    : duration_(0)
    , ffn_down_(std::move(ffn_down))
    , ffn_gate_(std::move(ffn_gate))
    , ffn_up_(std::move(ffn_up))
    , ffn_norm_(std::move(ffn_norm))
{
}
FeedForwardSwiGLU::~FeedForwardSwiGLU()
{
}

FeedForwardSwiGLU::FeedForwardSwiGLU(FeedForwardSwiGLU&& other)
    : duration_(0)
    , ffn_down_(std::move(other.ffn_down_))
    , ffn_gate_(std::move(other.ffn_gate_))
    , ffn_up_(std::move(other.ffn_up_))
    , ffn_norm_(std::move(other.ffn_norm_))
{
}

FeedForwardSwiGLU& FeedForwardSwiGLU::operator=(FeedForwardSwiGLU&& other)
{
    if(this != &other) {
        duration_ = 0;
        ffn_down_ = std::move(other.ffn_down_);
        ffn_gate_ = std::move(other.ffn_gate_);
        ffn_up_ = std::move(other.ffn_up_);
        ffn_norm_ = std::move(other.ffn_norm_);
    }
    return *this;
}

void FeedForwardSwiGLU::forward(
    const Config& config,
    Tensor& output,
    const Tensor& input,
        Tensor& buffer0,
    Tensor& buffer1)
{
    u64 dim = config.dimension_;
    u32 hidden_dim = static_cast<u32>(config.hidden_dim_);
    op::matmul(buffer0.data<f32>(), input.data<f32>(), ffn_gate_.data<f32>(), dim, hidden_dim);
    op::matmul(buffer1.data<f32>(), input.data<f32>(), ffn_up_.data<f32>(), dim, hidden_dim);

    // SwiGLU non-linearity
    for(u64 i = 0; i < hidden_dim; ++i) {
        f32 value = buffer0.data<f32>()[i];
        // silu(x) = x*σ(x) where σ(x) is thelogistic sigmoid
        value *= (1.0f / (1.0f + ::expf(-value)));
        value *= buffer1.data<f32>()[i];
        buffer0.data<f32>()[i] = value;
    }
    op::matmul(output.data<f32>(), buffer0.data<f32>(), ffn_down_.data<f32>(), hidden_dim, dim);
}

//--- TransformerBlock
//-----------------------------------------------------------
    TransformerBlock::TransformerBlock()
    :duration_(0)
{
}

    TransformerBlock::~TransformerBlock()
{
}

    TransformerBlock::TransformerBlock(TransformerBlock&& other)
{
}

TransformerBlock& TransformerBlock::operator=(TransformerBlock&& other)
{
    return *this;
}

void TransformerBlock::forward(
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
        Tensor& hbuffer1)
{
    attn_rmsnorm_.forward(buffer0, input);
    u64 kv_dim = (config.dimension_ * config.num_kv_heads_) / config.num_heads_;
    u64 layer_offset = layer * config.sequence_length_ * kv_dim;
    attn_.forward(
        config,
        position,
        layer_offset,
        buffer1,
        buffer0,
        query,
        key_cache,
        value_cache,
        attention);
    attn_residual_.forward(input, input, buffer1);
    ff_rmsnorm_.forward(buffer0, input);
    ff_.forward(
        config,
        buffer0,
        buffer0,
        hbuffer0,
        hbuffer1);
    ff_residual_.forward(output, input, buffer0);
}

//--- Vocabulary
//-----------------------------------------------------------
Vocabulary::Vocabulary()
    : model_{}
    , tokens_{}
    , scores_{}
    , token_types_{}
    , merges_{}
    , added_tokens_{}
    , bos_token_id_(1)
    , eos_token_id_(2)
    , unknown_token_id_(0)
    , separator_token_id_(-1)
    , padding_token_id_(-1)
    , cls_token_id_(-1)
    , mask_token_id_(-1)
    , add_bos_(-1)
    , add_eos_(-1)
    , linefeed_id_(13)
    , prefix_id_(-1)
    , suffix_id_(-1)
    , middle_id_(-1)
    , eot_id_(-1)
    , add_space_prefix_(true)
    , max_token_length_(0)
{
}

Vocabulary::Vocabulary(const gguf::GGUF& model_data)
    : model_{}
    , tokens_{}
    , scores_{}
    , token_types_{}
    , merges_{}
    , added_tokens_{}
    , bos_token_id_(1)
    , eos_token_id_(2)
    , unknown_token_id_(0)
    , separator_token_id_(-1)
    , padding_token_id_(-1)
    , cls_token_id_(-1)
    , mask_token_id_(-1)
    , add_bos_(-1)
    , add_eos_(-1)
    , linefeed_id_(13)
    , prefix_id_(-1)
    , suffix_id_(-1)
    , middle_id_(-1)
    , eot_id_(-1)
    , add_space_prefix_(true)
    , max_token_length_(0)
{
    using namespace gguf;
    const gguf_metadata_kv_t* metadata = nullptr;
    if(model_data.getMetaData(metadata, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_STRING, u8"tokenizer.ggml.model")) {
        model_ = model_data.getMetaDataString(*metadata);
    }

    if(model_data.getArrayMetaData(metadata, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_STRING, u8"tokenizer.ggml.tokens")) {
        tokens_ = model_data.getMetaDataArray(*metadata);
    }
    if(model_data.getArrayMetaData(metadata, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_FLOAT32, u8"tokenizer.ggml.scores")) {
        scores_ = model_data.getMetaDataArray(*metadata);
    }
    if(model_data.getArrayMetaData(metadata, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_INT32, u8"tokenizer.ggml.token_type")) {
        token_types_ = model_data.getMetaDataArray(*metadata);
    }
    if(model_data.getArrayMetaData(metadata, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_STRING, u8"tokenizer.ggml.merges")) {
        merges_ = model_data.getMetaDataArray(*metadata);
    }
    if(model_data.getArrayMetaData(metadata, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_STRING, u8"tokenizer.ggml.added_tokens")) {
        added_tokens_ = model_data.getMetaDataArray(*metadata);
    }

    if(model_data.getMetaData(metadata, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT32, u8"tokenizer.ggml.bos_token_id")) {
        bos_token_id_ = model_data.getMetaDataU32(*metadata);
    }
    if(model_data.getMetaData(metadata, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT32, u8"tokenizer.ggml.eos_token_id")) {
        eos_token_id_ = model_data.getMetaDataU32(*metadata);
    }
    if(model_data.getMetaData(metadata, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT32, u8"tokenizer.ggml.unknown_token_id")) {
        unknown_token_id_ = model_data.getMetaDataU32(*metadata);
    }
    if(model_data.getMetaData(metadata, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT32, u8"tokenizer.ggml.separator_token_id")) {
        separator_token_id_ = model_data.getMetaDataU32(*metadata);
    }
    if(model_data.getMetaData(metadata, gguf_metadata_value_type::GGUF_METADATA_VALUE_TYPE_UINT32, u8"tokenizer.ggml.padding_token_id")) {
        padding_token_id_ = model_data.getMetaDataU32(*metadata);
    }

    // Add tokens to maps
    tokenToId_.reserve(static_cast<u32>(tokens_.size_));
    idToToken_.resize(tokens_.size_);
    ::memset(&idToToken_[0], 0, sizeof(Token)*tokens_.size_);
    s32 id = 0;

    for(GGUFArray::Iterator<GGUFString> itr = tokens_.begin<GGUFString>(); itr; ++itr, ++id) {
        GGUFString ggufStr = *itr;
        String str = {ggufStr.length_, ggufStr.str_};
        max_token_length_ = (std::max)(max_token_length_, static_cast<s32>(ggufStr.length_));
        tokenToId_.add(str, id);
        if(id<idToToken_.size()){
            Token token = {};
            token.text_ = str;
            if(id<scores_.size_){
                token.score_ = scores_.get<f32>(id);
            }
            if(id<token_types_.size_){
                token.type_ = token_types_.get<s32>(id);
            }
            idToToken_[id] = token;
        }
    }
}

Vocabulary::~Vocabulary()
{
}

Vocabulary::Vocabulary(Vocabulary&& other)
    : model_(other.model_)
    , tokens_(other.tokens_)
    , scores_(other.scores_)
    , token_types_(other.token_types_)
    , merges_(other.merges_)
    , added_tokens_(other.added_tokens_)
    , bos_token_id_(other.bos_token_id_)
    , eos_token_id_(other.eos_token_id_)
    , unknown_token_id_(other.unknown_token_id_)
    , separator_token_id_(other.separator_token_id_)
    , padding_token_id_(other.padding_token_id_)
    , cls_token_id_(other.cls_token_id_)
    , mask_token_id_(other.mask_token_id_)
    , add_bos_(other.add_bos_)
    , add_eos_(other.add_eos_)
    , linefeed_id_(other.linefeed_id_)
    , prefix_id_(other.prefix_id_)
    , suffix_id_(other.suffix_id_)
    , middle_id_(other.middle_id_)
    , eot_id_(other.eot_id_)
    , add_space_prefix_(other.add_space_prefix_)
    , max_token_length_(other.max_token_length_)
    , tokenToId_(std::move(other.tokenToId_))
    , idToToken_(std::move(other.idToToken_))
{
    other.model_ = {};
    other.tokens_ = {};
    other.scores_ = {};
    other.token_types_ = {};
    other.merges_ = {};
    other.added_tokens_ = {};
    other.bos_token_id_ = 1;
    other.eos_token_id_ = 2;
    other.unknown_token_id_ = 0;
    other.separator_token_id_ = -1;
    other.cls_token_id_ = -1;
    other.mask_token_id_ = -1;
    other.add_bos_ = -1;
    other.add_eos_ = -1;
    other.linefeed_id_ = 13;
    other.prefix_id_ = -1;
    other.suffix_id_ = -1;
    other.middle_id_ = -1;
    other.eot_id_ = -1;
    other.add_space_prefix_ = true;
    other.max_token_length_ = 0;
}

Vocabulary& Vocabulary::operator=(Vocabulary&& other)
{
    if(this != &other) {
        model_ = other.model_;
        tokens_ = other.tokens_;
        scores_ = other.scores_;
        token_types_ = other.token_types_;
        merges_ = other.merges_;
        added_tokens_ = other.added_tokens_;
        bos_token_id_ = other.bos_token_id_;
        eos_token_id_ = other.eos_token_id_;
        unknown_token_id_ = other.unknown_token_id_;
        separator_token_id_ = other.separator_token_id_;
        cls_token_id_ = other.cls_token_id_;
        mask_token_id_ = other.mask_token_id_;
        add_bos_ = other.add_bos_;
        add_eos_ = other.add_eos_;
        linefeed_id_ = other.linefeed_id_;
        prefix_id_ = other.prefix_id_;
        suffix_id_ = other.suffix_id_;
        middle_id_ = other.middle_id_;
        eot_id_ = other.eot_id_;
        add_space_prefix_ = other.add_space_prefix_;
        max_token_length_ = other.max_token_length_;
        tokenToId_ = std::move(other.tokenToId_);
        idToToken_ = std::move(other.idToToken_);

        other.model_ = {};
        other.tokens_ = {};
        other.scores_ = {};
        other.token_types_ = {};
        other.merges_ = {};
        other.added_tokens_ = {};
        other.bos_token_id_ = 1;
        other.eos_token_id_ = 2;
        other.unknown_token_id_ = 0;
        other.separator_token_id_ = -1;
        other.cls_token_id_ = -1;
        other.mask_token_id_ = -1;
        other.add_bos_ = -1;
        other.add_eos_ = -1;
        other.linefeed_id_ = 13;
        other.prefix_id_ = -1;
        other.suffix_id_ = -1;
        other.middle_id_ = -1;
        other.eot_id_ = -1;
        other.max_token_length_ = 0;
    }
    return *this;
}

const gguf::GGUFString& Vocabulary::getModel() const
{
    return model_;
}

const gguf::GGUFArray& Vocabulary::getTokens() const
{
    return tokens_;
}

const gguf::GGUFArray& Vocabulary::getScores() const
{
    return scores_;
}

const gguf::GGUFArray& Vocabulary::getTokenTypes() const
{
    return token_types_;
}

const gguf::GGUFArray& Vocabulary::getMerges() const
{
    return merges_;
}

const gguf::GGUFArray& Vocabulary::getAddedTokens() const
{
    return added_tokens_;
}

s32 Vocabulary::getBOS() const
{
    return bos_token_id_;
}

s32 Vocabulary::getEOS() const
{
    return eos_token_id_;
}

s32 Vocabulary::getUnknown() const
{
    return unknown_token_id_;
}

s32 Vocabulary::getSeparator() const
{
    return separator_token_id_;
}

s32 Vocabulary::getPadding() const
{
    return padding_token_id_;
}

s32 Vocabulary::getCls() const
{
    return cls_token_id_;
}

s32 Vocabulary:: getMask() const
{
    return mask_token_id_;
}

s32 Vocabulary::getMaxTokenLength() const
{
    return max_token_length_;
}

bool Vocabulary::encode(s32& token, const char8_t* str) const
{
    assert(nullptr != str);
    String key;
    key.len_ = ::strnlen(reinterpret_cast<const char*>(str), 128);
    key.str_ = str;
    const s32* id = nullptr;
    if(tokenToId_.tryGet(key, id)) {
        token = *id;
        return true;
    }
    return false;
}

bool Vocabulary::encode(s32& token, u64 length, const char8_t* str) const
{
    assert(nullptr != str);
    String key;
    key.len_ = length;
    key.str_ = str;
    const s32* id = nullptr;
    if(tokenToId_.tryGet(key, id)) {
        token = *id;
        return true;
    }
    return false;
}

bool Vocabulary::decode(char8_t str[512], s32 token) const
{
    //if(static_cast<u64>(token)<idToToken_.size()){
    //    const String& value = idToToken_[static_cast<u32>(token)];
    //    u64 len = (std::min)(511ULL, value.len_);
    //    ::memcpy(str, value.str_, len);
    //    str[len] = u8'\0';
    //    return true;
    //}
    return false;
}

bool Vocabulary::decode(u64 length, char8_t str[], s32 token) const
{
    assert(0 < length);
    //if(static_cast<u64>(token)<idToToken_.size()){
    //    const String& value = idToToken_[static_cast<u32>(token)];
    //    u64 len = (std::min)(length - 1, value.len_);
    //    ::memcpy(str, value.str_, len);
    //    str[len] = u8'\0';
    //    return true;
    //}
    return false;
}

//--- Tokenizer
//-----------------------------------------------------------
const char8_t* Tokenizer::Pattern = u8R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
Tokenizer::Tokenizer()
    :buffer_(nullptr)
{
}

Tokenizer::Tokenizer(const gguf::GGUF& model_data)
    :buffer_(nullptr)
    ,tokens_(model_data)
{
    buffer_ = (char8_t*)allocate((tokens_.getMaxTokenLength() + 1 + 2)*sizeof(char8_t));
}

Tokenizer::~Tokenizer()
{
    deallocate(buffer_);
    buffer_ = nullptr;
}

Tokenizer::Tokenizer(Tokenizer&& other) noexcept
    :tokens_(std::move(other.tokens_))
    ,buffer_(other.buffer_)
{
    other.buffer_ = nullptr;
}

Tokenizer& Tokenizer::operator=(Tokenizer&& other)
{
    if(this != &other){
        deallocate(buffer_);
        tokens_ = std::move(other.tokens_);
        buffer_ = other.buffer_;

        other.buffer_ = nullptr;
    }
    return *this;
}

Array<s32> Tokenizer::tokenize(const std::string& text)
{
    Array<s32> result;
    symbols_.clear();
    symbols_.reserve(text.size());
    // split string into utf8 chars
        s32 index = 0;
        u64 offset = 0;
        while(offset < text.size()){
            Symbol symbol;
            size_t len = length(text[offset]);
            symbol.text_ = reinterpret_cast<const char8_t*>(text.c_str()) + offset;
            symbol.len_ = (std::min)(static_cast<u64>(len), text.size() - offset);
            offset += symbol.len_;
            symbol.prev_ = index - 1;
            symbol.next_ = offset == text.size() ? -1 : index + 1;
            ++index;
            symbols_.push_back(symbol);
        }

        // seed the work queue with all possible 2-character tokens.
        for (u64 i = 1; i < symbols_.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (0<work_queue_.size()) {
            Bigram bigram = work_queue_.front();
            work_queue_.pop_front();

            Symbol& left_sym = symbols_[bigram.left_];
            Symbol& right_sym = symbols_[bigram.right_];

            // if one of the symbols already got merged, skip it.
            if (left_sym.len_ == 0
                || right_sym.len_ == 0
                || (left_sym.len_ + right_sym.len_) != bigram.size_) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.len_ += right_sym.len_;
            right_sym.len_ = 0;

            // remove the right sym from the chain
            left_sym.next_ = right_sym.next_;
            if (0<=right_sym.next_) {
                symbols_[right_sym.next_].prev_ = bigram.left_;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev_, bigram.left_);
            try_add_bigram(bigram.left_, left_sym.next_);
        }

        for (s32 i = 0; i != -1; i = symbols_[i].next_) {
            Symbol& symbol = symbols_[i];
            resegment(symbol, output);
        }
        return result;
}

u64 Tokenizer::length(char c)
{
    static const u64 lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    return lookup[static_cast<uint8_t>(c) >> 4];
}

void Tokenizer::try_add_bigram(Symbol::index left, Symbol::index right)
{
    if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(reinterpret_cast<const char*>(symbols_[left].text_), symbols_[left].len_ + symbols_[right].len_);
        auto token = vocab.token_to_id.find(text);

        if (token == vocab.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab.id_to_token.size()) {
            return;
        }

        const auto & tok_data = vocab.id_to_token[(*token).second];

        Bigram bigram;
        bigram.left  = left;
        bigram.right = right;
        bigram.score = tok_data.score;
        bigram.size  = text.size();

        work_queue.push(bigram);

        // Do we need to support is_unused?
        rev_merge[text] = std::make_pair(left, right);
}

//--- Sampler
//-----------------------------------------------------------
Sampler::Sampler()
    : vocab_size_(0)
    , temperature_(1.0f)
    , topp_(0.9f)
{
}

Sampler::~Sampler()
{
}

Sampler::Sampler(Sampler&& other)
{
}

Sampler& Sampler::operator=(Sampler&& other)
{
    return *this;
}

u32 Sampler::sample(f32* logits)
{
    assert(nullptr != logits);
    if(temperature_<=std::numeric_limits<f32>::epsilon()){
        return sample_argmax(vocab_size_, logits);
    }
    f32 inv_temperature = 1.0f/temperature_;
    for(u32 i=0; i<vocab_size_; ++i){
        logits[i] *= inv_temperature;
    }
    op::softmax(vocab_size_, logits);
    f32 coin = random_.frand();
    if(topp_<=0.0f || 1.0f<=topp_){
        return sample_mult(vocab_size_, coin, logits);
    }else{
        return sample_topp(vocab_size_, topp_, coin, probindex_, logits);
    }
}

u32 Sampler::sample_argmax(u32 size, const f32* probabilities)
{
    assert(0 < size);
    u32 max_index = 0;
    f32 max_p = probabilities[0];
    for(u32 i = 1; i < size; ++i) {
        if(max_p < probabilities[i]) {
            max_index = i;
            max_p = probabilities[i];
        }
    }
    return max_index;
}

u32 Sampler::sample_mult(u32 size, f32 coin, const f32* probabilities)
{
    f32 cdf = 0.0f;
    for(u32 i=0; i<size; ++i){
        cdf += probabilities[i];
        if(coin < cdf){
            return i;
        }
    }
    return size-1;
}

u32 Sampler::sample_topp(u32 size, f32 topp, f32 coin, ProbIndex* probindex, const f32* probabilities)
{
    u32 n0 = 0;
    const f32 cutoff = (1.0f-topp)/(size-1);
    for(u32 i=0; i<size; ++i){
        if(cutoff<=probabilities[i]){
            probindex[n0].index_ = i;
            probindex[n0].prob_ = probabilities[i];
            ++n0;
        }
    }
    std::sort(probindex, probindex + n0, [](const ProbIndex& x0, const ProbIndex& x1) {
        return x0.prob_ > x1.prob_;
    });
    f32 cumulative_prob = 0.0f;
    u32 end = n0;
    for(u32 i=0; i<n0; ++i){
        cumulative_prob += probindex[i].prob_;
        if(topp<cumulative_prob){
            end = i+1;
            break;
        }
    }
    f32 r = coin * cumulative_prob;
    f32 cdf = 0.0f;
    for(u32 i=0; i<end; ++i){
        cdf += probindex[i].prob_;
        if(r<cdf){
            return probindex[i].index_;
        }
    }
    return probindex[end-1].index_;
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

Llama2::Llama2(Llama2&& /*other*/)
{
}

Llama2::~Llama2()
{
}

Llama2& Llama2::operator=(Llama2&& other)
{
    if(this != &other) {
        config_ = other.config_;
    }
    return *this;
}

void Llama2::forward(u32 token, u32 position)
{
    for(u64 i = 0; i < config_.num_layers_; ++i) {
        blocks_[i].forward(
            config_,
            i,
            position,
            context_.x_,
            context_.x_,
            context_.query_,
            context_.key_cache_,
            context_.value_cache_,
            context_.attn_,
            context_.xb_,
            context_.xb2_,
            context_.hb_,
            context_.hb2_);
    }
    output_rmsnorm_.forward(context_.x_, context_.x_);
    op::matmul(
        context_.logits_.data<f32>(),
        context_.x_.data<f32>(),
        output_weight_.data<f32>(),
        config_.dimension_,
        config_.vocab_size_);
}
} // namespace cppgpt

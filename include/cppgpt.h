#ifndef INC_CPPGPT_H_
#define INC_CPPGPT_H_
#include <cstdint>
#include <cassert>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

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

//--- Module
//-----------------------------------------------------------
class Module
{
public:
protected:
	Module();
	virtual ~Module();
	virtual dnnl::memory forward(dnnl::memory tensor, uint32_t start_pos, dnnl::memory freqs_cis, dnnl::memory mask);
private:
	Module(const Module&) = delete;
	Module& operator=(const Module&) = delete;
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
class Llama2 : public virtual Model
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
};
} // namespace cppgpt
#endif // INC_CPPGPT_H_

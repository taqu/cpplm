#include "cppgpt.h"

namespace cppgpt
{
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
    :config_(config)
{
}

Llama2::Llama2(Llama2&& other)
    :network_(std::move(other.network_))
{
}

Llama2::~Llama2()
{
	network_.clear();
}

Llama2& Llama2::operator=(Llama2&& other)
{
}
} // namespace cppgpt

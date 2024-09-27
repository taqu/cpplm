#include "catch_amalgamated.hpp"
#include <vector>
#include <random>
#include "gguf.h"

TEST_CASE("Load GGUF" "[GGUF]")
{
	using namespace gguf;
	GGUF gguf;
	//gguf::Error result = gguf.load(u8"./data/tinyllama-1.1b-chat-v1.0.Q2_K.gguf");
	gguf::Error result = gguf.load(u8"./data/gpt2.Q8_0.gguf");
	CHECK(gguf::Error::Success == result);
}

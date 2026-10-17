#pragma once

#include "ggml.h"

struct ggml_cannge_build_context;

bool ggml_cannge_build_rope(ggml_cannge_build_context & ctx, ggml_tensor * node);

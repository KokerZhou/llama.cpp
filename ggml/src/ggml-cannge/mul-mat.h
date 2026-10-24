#pragma once

#include "ggml.h"
#include <ge/es_graph_builder.h>

struct ggml_cannge_build_context;

bool ggml_cannge_build_matmul_2d(ggml_cannge_build_context & ctx,
        const ge::es::EsTensorHolder & a, const ge::es::EsTensorHolder & b, ge::es::EsTensorHolder & output);

bool ggml_cannge_build_matmul(ggml_cannge_build_context & ctx, ggml_tensor * node);

/*
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "ops.h"
#include "graph-build-priv.h"
#include "elementwise.h"
#include "norm.h"
#include "tensor.h"
#include "rope.h"
#include "mul-mat.h"
#include "fattn.h"

bool ggml_cannge_build_node(ggml_cannge_build_context & ctx, ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SCALE:
        case GGML_OP_UNARY:
        case GGML_OP_GLU:
            return ggml_cannge_build_elementwise(ctx, node);

        case GGML_OP_SOFT_MAX:
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
            return ggml_cannge_build_norm(ctx, node);

        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_VIEW:
        case GGML_OP_CONT:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_GET_ROWS:
        case GGML_OP_CONCAT:
        case GGML_OP_SET_ROWS:
            return ggml_cannge_build_tensor(ctx, node);

        case GGML_OP_ROPE:
            return ggml_cannge_build_rope(ctx, node);

        case GGML_OP_MUL_MAT:
            return ggml_cannge_build_matmul(ctx, node);

        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cannge_build_attention(ctx, node);

        default:
            return ctx.fail("unsupported op " + std::to_string((int) node->op));
    }
}

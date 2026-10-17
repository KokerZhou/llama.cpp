#pragma once

#include "ggml-backend.h"
#include "options.h"

bool ggml_cannge_float_dtype(enum ggml_type type);
bool ggml_backend_cannge_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);
bool ggml_backend_cannge_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op);

// On rejection, reason receives a static string; on success it receives nullptr.
bool ggml_cannge_supports_op(const ggml_tensor * op, const ggml_cannge_options & options, const char ** reason = nullptr);

// Shared shape and GQA policy for the Mul/ReduceSum path; callers validate types and contraction size.
bool ggml_cannge_use_gemv(const ggml_tensor * weights, const ggml_tensor * input, const ggml_cannge_options & options);

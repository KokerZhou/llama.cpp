#pragma once

#include "ggml-backend.h"

bool ggml_cannge_float_dtype(enum ggml_type type);
bool ggml_backend_cannge_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);
bool ggml_backend_cannge_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op);

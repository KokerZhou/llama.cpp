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

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Maximum number of CANNGE devices supported.
 */
#define GGML_CANNGE_MAX_DEVICES 16

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cannge_reg(void);

/**
 * @brief Initializes the CANNGE backend for a specified device.
 *
 * @param device The index of the device to initialize.
 * @return A pointer to the initialized backend instance, or nullptr on failure.
 */
GGML_BACKEND_API ggml_backend_t ggml_backend_cannge_init(int32_t device);

/**
 * @brief Checks if a given backend is a CANNGE backend.
 *
 * @param backend The backend instance to check.
 * @return True if the backend is a CANNGE backend, false otherwise.
 */
GGML_BACKEND_API bool ggml_backend_is_cannge(ggml_backend_t backend);

/**
 * @brief Retrieves the CANNGE buffer type for a specified device.
 *
 * @param device The device index.
 * @return A pointer to the buffer type interface for the specified device, or
 * nullptr if the device index is out of range.
 */
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cannge_buffer_type(int32_t device);

/**
 * @brief Retrieves the number of CANNGE devices.
 *
 * @return The number of available CANNGE devices.
 */
GGML_BACKEND_API int32_t ggml_backend_cannge_get_device_count(void);

/**
 * @brief Retrieves the description of a CANNGE device.
 *
 * @param device The device index.
 * @param description Buffer to store the device description.
 * @param description_size Size of the description buffer.
 */
GGML_BACKEND_API void ggml_backend_cannge_get_device_description(int32_t device, char * description, size_t description_size);

/**
 * @brief Retrieves the memory information of a CANNGE device.
 *
 * @param device The device index.
 * @param free Pointer to store the free memory in bytes.
 * @param total Pointer to store the total memory in bytes.
 */
GGML_BACKEND_API void ggml_backend_cannge_get_device_memory(int32_t device, size_t * free, size_t * total);

#ifdef __cplusplus
}
#endif

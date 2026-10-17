#pragma once

#include "common.h"

const void * ggml_cannge_staging_source(const ggml_tensor * tensor);
bool ggml_cannge_strided_copy(const ggml_tensor * t, void * dense, bool dense_to_strided, aclrtStream stream);
// Pending uploads must drain even when the existing capacity is sufficient.
bool ggml_cannge_reserve_input_arena(void *& device, void *& host, size_t & capacity, size_t size, aclrtStream stream, bool pending);
struct ggml_cannge_set_rows_op;
bool ggml_cannge_set_rows(const ggml_cannge_set_rows_op & op, const std::vector<int64_t> & slots,
                          const std::vector<char> & data, aclrtStream stream);

bool ggml_cannge_read_indices(const ggml_tensor * tensor, aclrtStream stream, std::vector<int64_t> & slots);
bool ggml_cannge_upload_attention_length(const ggml_tensor * mask, void * destination, aclrtStream stream);

// Blocking snapshot: device reads first drain the producer stream.
bool ggml_cannge_read_snapshot(const ggml_tensor * tensor, size_t size, aclrtStream stream, std::vector<char> & snapshot);

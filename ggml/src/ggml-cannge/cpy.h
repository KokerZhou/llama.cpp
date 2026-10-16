#pragma once

#include "common.h"

const void * ggml_cannge_staging_source(const ggml_tensor * tensor);
bool ggml_cannge_strided_copy(const ggml_tensor * t, void * dense, bool dense_to_strided, aclrtStream stream);
bool ggml_cannge_reserve_input_arena(void *& device, void *& host, size_t & capacity, size_t size, aclrtStream stream);
struct ggml_cannge_set_rows_op;
bool ggml_cannge_set_rows(const ggml_cannge_set_rows_op & op, const std::vector<int64_t> & slots,
                          const std::vector<char> & data, aclrtStream stream);

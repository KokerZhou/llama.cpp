#pragma once

#include <cstdint>
#include <string>

// One immutable-by-convention snapshot for support checks, analysis and graph construction.
struct ggml_cannge_options {
    bool disable_all = false;
    bool disable_binary = false;
    bool disable_softmax = false;
    bool disable_unary = false;
    bool disable_rms = false;
    bool disable_matmul = false;
    bool disable_get_rows = false;
    bool disable_rope = false;
    bool flash_attention = false;
    bool host_buffers = false;
    bool limit_matmul = false;
    bool contiguous_views = false;         // GGML_CANNGE_CONT_NC
    bool disable_contiguous_views = false; // GGML_CANNGE_NO_CONT_NC
    bool disable_splice = false;
    bool scatter = false;
    bool scatter_add_zero = false;
    bool attention_input_permute = false;
    bool attention_output_permute = false;
    bool attention_ifa_all = false;
    bool gemv_f16 = true;
    bool gqa_present = false;
    std::string gqa_mode;
    bool dump_present = false;
    std::string dump_filter;
    std::string attention_layout = "BSND";
    int64_t attention_pre_tokens = 2147483647;
    int64_t attention_next_tokens = 2147483647;
    int64_t attention_sequence_length = 0;

    static ggml_cannge_options from_env();
    std::string signature() const;
};

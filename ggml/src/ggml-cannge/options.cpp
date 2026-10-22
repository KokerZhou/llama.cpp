#include "options.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int64_t ggml_cannge_integer_option(const char * name, int64_t fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr) {
        return fallback;
    }
    char * end = nullptr;
    errno = 0;
    const int64_t parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0' || errno == ERANGE) {
        std::fprintf(stderr, "CANNGE: invalid integer for %s; using default %lld\n", name, (long long) fallback);
        return fallback;
    }
    return parsed;
}

ggml_cannge_options ggml_cannge_options::from_env() {
    ggml_cannge_options options;
    options.disable_all = std::getenv("GGML_CANNGE_DISABLE_ALL") != nullptr;
    options.disable_binary = std::getenv("GGML_CANNGE_NO_ADDMUL") != nullptr;
    options.disable_softmax = std::getenv("GGML_CANNGE_NO_SOFTMAX") != nullptr;
    options.disable_unary = std::getenv("GGML_CANNGE_NO_UNARY") != nullptr;
    options.disable_rms = std::getenv("GGML_CANNGE_NO_RMS") != nullptr;
    options.disable_matmul = std::getenv("GGML_CANNGE_NO_MATMUL") != nullptr;
    options.disable_get_rows = std::getenv("GGML_CANNGE_NO_GET_ROWS") != nullptr;
    options.disable_rope = std::getenv("GGML_CANNGE_NO_ROPE") != nullptr;
    options.flash_attention = std::getenv("GGML_CANNGE_FA") != nullptr;
    options.host_buffers = std::getenv("GGML_CANNGE_HOST_BUFT") != nullptr;
    options.limit_matmul = std::getenv("GGML_CANNGE_BIGMM") != nullptr;
    options.contiguous_views = std::getenv("GGML_CANNGE_CONT_NC") != nullptr;
    options.disable_contiguous_views = std::getenv("GGML_CANNGE_NO_CONT_NC") != nullptr;
    options.disable_splice = std::getenv("GGML_CANNGE_NO_SPLICE") != nullptr;
    options.scatter = std::getenv("GGML_CANNGE_SCATTER") != nullptr &&
                      std::getenv("GGML_CANNGE_NO_SCATTER") == nullptr;
    options.scatter_add_zero = std::getenv("GGML_CANNGE_SR_ADD0") != nullptr;
    options.gemv_bmm = std::getenv("GGML_CANNGE_BMM_GEMV") != nullptr;
    options.attention_input_permute = std::getenv("GGML_CANNGE_FA_INPERM") != nullptr;
    options.attention_output_permute = std::getenv("GGML_CANNGE_FA_OUTPERM") != nullptr;
    options.attention_ifa_all = std::getenv("GGML_CANNGE_FA_IFA_ALL") != nullptr;
    const char * gemv = std::getenv("GGML_CANNGE_GEMV16");
    options.gemv_f16 = gemv == nullptr || std::strcmp(gemv, "0") != 0;
    const char * gqa = std::getenv("GGML_CANNGE_GQA");
    options.gqa_present = gqa != nullptr;
    options.gqa_mode = gqa != nullptr ? gqa : "";
    const char * dump = std::getenv("GGML_CANNGE_DUMP");
    options.dump_present = dump != nullptr;
    options.dump_filter = dump != nullptr ? dump : "";
    const char * layout = std::getenv("GGML_CANNGE_FA_LAYOUT");
    options.attention_layout = layout != nullptr ? layout : "BSND";
    options.attention_pre_tokens = ggml_cannge_integer_option("GGML_CANNGE_FA_PRE", 2147483647);
    options.attention_next_tokens = ggml_cannge_integer_option("GGML_CANNGE_FA_NEXT", 2147483647);
    options.attention_sequence_length = ggml_cannge_integer_option("GGML_CANNGE_FA_ASL", 0);
    return options;
}

std::string ggml_cannge_options::signature() const {
    std::string result;
    auto boolean = [&](bool value) { result.push_back(value ? 1 : 0); };
    auto integer = [&](int64_t value) { result.append(reinterpret_cast<const char *>(&value), sizeof(value)); };
    auto string = [&](const std::string & value) {
        const uint64_t size = value.size();
        result.append(reinterpret_cast<const char *>(&size), sizeof(size));
        result.append(value);
    };
    boolean(disable_all);
    boolean(disable_binary);
    boolean(disable_softmax);
    boolean(disable_unary);
    boolean(disable_rms);
    boolean(disable_matmul);
    boolean(disable_get_rows);
    boolean(disable_rope);
    boolean(flash_attention);
    boolean(host_buffers);
    boolean(limit_matmul);
    boolean(contiguous_views);
    boolean(disable_contiguous_views);
    boolean(disable_splice);
    boolean(scatter);
    boolean(scatter_add_zero);
    boolean(gemv_bmm);
    boolean(attention_input_permute);
    boolean(attention_output_permute);
    boolean(attention_ifa_all);
    boolean(gemv_f16);
    boolean(gqa_present);
    string(gqa_mode);
    boolean(dump_present);
    string(dump_filter);
    string(attention_layout);
    integer(attention_pre_tokens);
    integer(attention_next_tokens);
    integer(attention_sequence_length);
    return result;
}

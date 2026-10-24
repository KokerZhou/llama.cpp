#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <memory>
static std::vector<llama_token> tokens(const llama_vocab * vocab, const char * text, bool special) {
    int n = -llama_tokenize(vocab, text, std::strlen(text), nullptr, 0, special, true);
    if (n <= 0) std::exit(2);
    std::vector<llama_token> out(n);
    if (llama_tokenize(vocab, text, std::strlen(text), out.data(), n, special, true) != n) std::exit(2);
    return out;
}
int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s MODEL CPU|CANNGE0 OUTPUT\n", argv[0]);
        return 2;
    }
    const bool device = std::strcmp(argv[2], "0") != 0 && std::strcmp(argv[2], "CPU") != 0;
    const char * device_name = std::strcmp(argv[2], "1") == 0 ? "CANNGE0" : argv[2];
    ggml_backend_load_all();
    llama_backend_init();
    struct backend_guard { ~backend_guard() { llama_backend_free(); } } guard;
    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = {device ? ggml_backend_dev_by_name(device_name) : nullptr, nullptr};
    if (device && !devices[0]) return 3;
    mp.devices = devices;
    mp.n_gpu_layers = device ? 99 : 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) return 4;
    auto cp = llama_context_default_params();
    cp.n_ctx = 128; cp.n_batch = 64; cp.n_ubatch = 64;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv = cp.op_offload = device;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) return 5;
    const auto * vocab = llama_model_get_vocab(model.get());
    auto prompt = tokens(vocab, "The capital of France is", true);
    auto continuation = tokens(vocab, " Paris. The capital of Japan is Tokyo.", false);
    const int32_t header[] = {llama_vocab_n_tokens(vocab), int32_t(continuation.size() + 1), 2};
    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(argv[3], "wb"), std::fclose);
    if (!file || std::fwrite(header, sizeof(header), 1, file.get()) != 1) return 6;
    for (int round = 0; round < 2; ++round) {
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        for (size_t step = 0; step <= continuation.size(); ++step) {
            auto batch = step == 0 ? llama_batch_get_one(prompt.data(), prompt.size()) : llama_batch_get_one(&continuation[step-1], 1);
            const int status = llama_decode(ctx.get(), batch);
            if (status != 0) { std::fprintf(stderr, "decode failed round=%d step=%zu status=%d\n", round, step, status); return 7; }
            const float * logits = llama_get_logits_ith(ctx.get(), -1);
            if (!logits || std::fwrite(logits, sizeof(float), header[0], file.get()) != size_t(header[0])) return 8;
            std::fprintf(stderr, "CHECK round=%d step=%zu top1=%td\n", round, step, std::max_element(logits, logits+header[0])-logits);
        }
    }
    return std::fclose(file.release()) == 0 ? 0 : 6;
}

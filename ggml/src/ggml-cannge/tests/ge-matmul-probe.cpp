#include <ge/ge_api.h>
#include <ge/ge_error_codes.h>
#include <ge/es_graph_builder.h>
#include <es_nn/es_nn_ops.h>
#include <es_math/es_math_ops.h>
#include <graph/tensor.h>
#include <acl/acl.h>
#include <cmath>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

static void check(ge::Status status, const char * phase) {
    std::fprintf(stderr, "%s status=%u\n", phase, unsigned(status));
    if (status != ge::SUCCESS) {
        std::fprintf(stderr, "%s\n", ge::GEGetErrorMsg().c_str());
        std::fflush(nullptr);
        std::_Exit(90);
    }
}

int main(int argc, char ** argv) {
    const int64_t M = 16;
    int64_t N = 4096, K = 2560;
    const bool matmul = argc > 1 && std::strcmp(argv[1], "matmul") == 0;
    bool valid = argc == 1 || (argc >= 2 && (matmul || std::strcmp(argv[1], "bmm") == 0));
    if (argc == 3 && std::strcmp(argv[2], "small") == 0) {
        N = 16; K = 256;
    } else if (argc == 4) {
        auto dimension = [](const char * text, int64_t & value) {
            const char * end = text + std::strlen(text);
            const auto result = std::from_chars(text, end, value);
            return result.ec == std::errc() && result.ptr == end && value > 0 && value <= 8192;
        };
        valid = valid && dimension(argv[2], N) && dimension(argv[3], K);
    } else if (argc > 2) {
        valid = false;
    }
    if (!valid || N * K > 32 * 1024 * 1024) {
        std::fprintf(stderr, "usage: %s [bmm|matmul [small|N K]]; dimensions 1..8192, N*K <= 33554432\n", argv[0]);
        return 2;
    }
    std::fprintf(stderr, "probe op=%s M=%lld N=%lld K=%lld\n", matmul ? "MatMulV3" : "BatchMatMulV3",
                 (long long) M, (long long) N, (long long) K);
    if (aclInit(nullptr) != 0 || aclrtSetDevice(0) != 0) return 2;
    std::map<ge::AscendString, ge::AscendString> options = {
        {"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "0"},
        {"ge.exec.precision_mode", "allow_fp32_to_fp16"},
    };
    check(ge::GEInitialize(options), "initialize");
    {
        std::map<ge::AscendString, ge::AscendString> session_options = {{"ge.session_device_id", "0"}};
        ge::Session session(session_options);
        ge::es::EsGraphBuilder builder("projection");
        auto a = builder.CreateInput(0, "a", ge::DT_FLOAT16, ge::FORMAT_ND, {M, K});
        auto b = builder.CreateInput(1, "b", ge::DT_FLOAT16, ge::FORMAT_ND, {N, K});
        auto y = matmul ? ge::es::MatMulV3(a, b, nullptr, nullptr, false, true)
                        : ge::es::BatchMatMulV3(a, b, nullptr, nullptr, false, true);
        auto output = ge::es::Cast(y, int64_t(ge::DT_FLOAT));
        check(ge::es::EsGraphBuilder::SetOutput(output, 0), "set output");
        auto graph = builder.BuildAndReset();
        if (!graph) return 3;
        check(session.AddGraphWithCopy(1, *graph), "add graph");
        std::vector<uint16_t> ah(M*K, 0x3c00), bh(N*K, 0x3c00);
        std::vector<ge::Tensor> inputs;
        inputs.emplace_back(ge::TensorDesc(ge::Shape({M,K}), ge::FORMAT_ND, ge::DT_FLOAT16),
                            reinterpret_cast<const uint8_t *>(ah.data()), ah.size()*sizeof(uint16_t));
        inputs.emplace_back(ge::TensorDesc(ge::Shape({N,K}), ge::FORMAT_ND, ge::DT_FLOAT16),
                            reinterpret_cast<const uint8_t *>(bh.data()), bh.size()*sizeof(uint16_t));
        std::vector<ge::Tensor> outputs;
        check(session.RunGraph(1, inputs, outputs), "run graph");
        if (outputs.size() != 1 || outputs[0].GetSize() != M*N*sizeof(float)) return 4;
        for (int64_t i = 0; i < M*N; ++i) {
            float value;
            std::memcpy(&value, outputs[0].GetData()+i*sizeof(float), sizeof(value));
            if (!std::isfinite(value) || std::abs(value-float(K)) > 0.01f) {
                std::fprintf(stderr, "mismatch at %lld: %g expected %lld\n", (long long)i, value, (long long)K);
                std::_Exit(5);
            }
        }
        check(session.RemoveGraph(1), "remove graph");
    }
    check(ge::GEFinalize(), "finalize");
    if (aclrtResetDevice(0) != 0 || aclFinalize() != 0) return 6;
    std::printf("PASS: all %lld outputs equal %lld\n", (long long)(M*N), (long long)K);
}

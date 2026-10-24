#include "mul-mat.h"
#include "graph-build-priv.h"

bool ggml_cannge_build_matmul_2d(ggml_cannge_build_context & ctx,
        const cannge_es::EsTensorHolder & a, const cannge_es::EsTensorHolder & b,
        cannge_es::EsTensorHolder & output) {
    auto * left = a.GetCTensorHolder();
    auto * right = b.GetCTensorHolder();
    auto * builder = ctx.builder.GetCGraphBuilder();
    if (left == nullptr || right == nullptr || builder == nullptr || builder->GetGraph() == nullptr ||
        &left->GetOwnerBuilder() != builder || &right->GetOwnerBuilder() != builder) {
        return ctx.fail("MatMulV2 inputs must belong to the current graph");
    }
    // Match the public graph prototype. ES assembly can run before GE loads the operator registry.
    using node_builder = cannge_es::CompliantNodeBuilder;
    auto name = builder->GenerateNodeName("MatMulV2");
    auto & graph = *builder->GetGraph();
    auto node = node_builder(&graph)
            .OpType("MatMulV2").Name(name.GetString())
            .IrDefInputsV2({{"x1", node_builder::kEsIrInputRequired, ""},
                            {"x2", node_builder::kEsIrInputRequired, ""},
                            {"bias", node_builder::kEsIrInputOptional, ""},
                            {"offset_w", node_builder::kEsIrInputOptional, ""}})
            .IrDefOutputsV2({{"y", node_builder::kEsIrOutputRequired, ""}})
            .IrDefAttrsV2({{"transpose_x1", node_builder::kEsAttrOptional, "VT_BOOL", cannge_es::CreateFrom(false)},
                           {"transpose_x2", node_builder::kEsAttrOptional, "VT_BOOL", cannge_es::CreateFrom(true)},
                           {"offset_x", node_builder::kEsAttrOptional, "VT_INT", cannge_es::CreateFrom(int64_t(0))}})
            .Build();
    for (int i = 0; i < 2; ++i) {
        auto * input = i == 0 ? left : right;
        const auto status = cannge_es::AddEdgeAndUpdatePeerDesc(
                graph, input->GetProducer(), input->GetOutIndex(), node, i);
        if (status != ge::GRAPH_SUCCESS) {
            return ctx.fail("MatMulV2 input[" + std::to_string(i) + "]: status " + std::to_string(status));
        }
    }
    auto * holder = builder->GetTensorHolderFromNode(node, 0);
    if (holder == nullptr) {
        return ctx.fail("MatMulV2 output is unavailable");
    }
    output = cannge_es::EsTensorHolder(holder);
    return true;
}

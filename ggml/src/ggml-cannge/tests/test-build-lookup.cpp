#include "graph-build-priv.h"
#include "tensor.h"
#include "mul-mat.h"
#include "supports.h"
#include "ggml-impl.h"
#include "ggml-cannge-impl.h"

#include <cstdio>
#include <cstdlib>

const char * ggml_type_name(ggml_type) { return "test-type"; }
// Assembly cases use only external inputs and allocate no plan buffers.
ggml_cannge_plan::~ggml_cannge_plan() = default;
bool ggml_cannge_build_node(ggml_cannge_build_context &, ggml_tensor *) { std::abort(); }
static void check(bool result, const char * message) {
    if (!result) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
static ggml_tensor tensor() {
    ggml_tensor t{};
    t.type = GGML_TYPE_F32;
    t.ne[0] = 4; t.ne[1] = 2; t.ne[2] = t.ne[3] = 1;
    t.nb[0] = 4; t.nb[1] = 16; t.nb[2] = t.nb[3] = 32;
    return t;
}
int main() {
    for (int rank : {2, 3, 4}) {
        ggml_cannge_plan_io matmul_io;
        std::string matmul_error;
        ggml_cannge_build_context matmul_ctx(matmul_io, matmul_error);
        auto weights = tensor(), input = tensor(), output = tensor();
        weights.ne[0] = weights.ne[1] = input.ne[0] = 16;
        input.ne[1] = 8;
        if (rank >= 3) { weights.ne[2] = input.ne[2] = 2; }
        if (rank == 4) { weights.ne[3] = input.ne[3] = 2; }
        output.op = GGML_OP_MUL_MAT;
        output.src[0] = &weights; output.src[1] = &input;
        matmul_ctx.tensors[&weights] = {1}; matmul_ctx.tensors[&input] = {2};
        ge::es::concat_calls = ge::es::slice_calls = 0;
        check(ggml_cannge_build_matmul(matmul_ctx, &output), "build F32 matmul with M=8");
        check(ge::es::concat_calls == (rank == 2 ? 1 : 0), "only rank-2 matmul pads M");
        check(ge::es::slice_calls == (rank == 2 ? 1 : 0), "only rank-2 matmul slices M");
        if (rank == 2) {
            check(ge::es::last_slice_size == std::vector<int64_t>({8, 16}), "padding restores output shape");
        }
    }
    using result = ggml_cannge_build_result;
    ggml_cannge_plan_io io;
    std::string error = "prior diagnostic";
    ggml_cannge_build_context ctx(io, error);
    auto base = tensor();
    cannge_es::EsTensorHolder holder{99};
    check(ctx.try_get(&base, holder) == result::not_applicable, "absent optional tensor");
    check(error == "prior diagnostic" && holder.id == 99, "absence preserves error and output");
    check(!ctx.get(&base, holder) && error.find("no ES tensor") != std::string::npos, "required lookup reports absence");
    ctx.tensors[&base] = {1};
    check(ctx.try_get(&base, holder) == result::success && holder.id == 1, "registered holder lookup");

    auto view = tensor();
    view.view_src = &base;
    view.op = GGML_OP_ADD; // A valid view chain with an unsupported relabel operation.
    holder.id = 99;
    check(ctx.try_get(&view, holder) == result::failure, "relabel failure is not absence");
    check(error.find("non-relabel") != std::string::npos && holder.id == 99, "failure retains diagnostic and output");
    check(ctx.tensors.count(&view) == 0, "failed lookup publishes no holder");
    auto cont = tensor();
    cont.op = GGML_OP_CONT; cont.src[0] = &view;
    check(ggml_cannge_build_cont_chain(ctx, &cont, holder) == result::failure, "CONT propagates source reconstruction failure");
    check(holder.id == 99, "failed CONT preserves output");

    view.op = GGML_OP_RESHAPE;
    check(ctx.try_get(&view, holder) == result::success, "supported view reconstruction");
    ctx.tensors.erase(&view);
    check(ggml_cannge_build_cont_chain(ctx, &cont, holder) == result::success,
          "CONT reconstructs an unregistered source from its base");
    check(ctx.tensors.count(&view) == 1, "CONT shares the lookup reconstruction cache");
    ctx.tensors.erase(&view);
    ctx.tensors.erase(&base);
    holder.id = 99; error = "prior diagnostic";
    check(ggml_cannge_build_cont_chain(ctx, &cont, holder) == result::not_applicable,
          "CONT without a registered base is unavailable");
    check(holder.id == 99 && error == "prior diagnostic", "unavailable CONT preserves output and error");
    ctx.tensors[&base] = {1};
    check(ggml_cannge_build_cont_chain(ctx, &cont, holder) == result::success,
          "CONT becomes available after registering its base");

    auto rows = tensor();
    rows.view_src = &base; rows.op = GGML_OP_ADD;
    auto indices = tensor();
    indices.type = GGML_TYPE_I32; indices.ne[0] = indices.ne[1] = 1;
    io.set_rows.push_back({&base, &rows, &indices, false});
    holder.id = 99;
    check(ggml_cannge_build_cont_chain(ctx, &cont, holder) == result::failure, "splice propagates row reconstruction failure");
    check(holder.id == 99 && error.find("non-relabel") != std::string::npos, "splice failure preserves output and reason");
    rows.view_src = nullptr;
    error = "prior diagnostic";
    check(ggml_cannge_build_cont_chain(ctx, &cont, holder) == result::success, "absent optional rows allow plain chain");
    check(error == "prior diagnostic", "optional splice preserves prior diagnostic");

    ctx.tensors[&rows] = {10000};
    indices.view_src = &base; indices.op = GGML_OP_ADD;
    holder.id = 99;
    check(ggml_cannge_build_cont_chain(ctx, &cont, holder) == result::failure, "splice propagates index reconstruction failure");
    check(holder.id == 99, "failed index lookup preserves CONT output");
    indices.view_src = nullptr;
    check(ggml_cannge_build_cont_chain(ctx, &cont, holder) == result::success, "absent optional indices allow plain chain");
    auto scatter = tensor();
    scatter.op = GGML_OP_SET_ROWS;
    scatter.src[0] = &rows; scatter.src[1] = &indices; scatter.src[2] = &base;
    io.options.scatter_add_zero = true;
    ctx.tensors.erase(&rows);
    rows.view_src = &base; rows.op = GGML_OP_ADD;
    check(!ggml_cannge_build_tensor(ctx, &scatter), "scatter materialization propagates lookup failure");
    rows.view_src = nullptr;
    check(ggml_cannge_build_tensor(ctx, &scatter), "scatter materialization allows absent optional rows");
    ctx.tensors[&rows] = {10000};
    check(ggml_cannge_build_tensor(ctx, &scatter) && ctx.tensors[&rows].id != 10000,
          "scatter materializes available rows");
    check(ctx.try_get(nullptr, holder) == result::failure, "null source is a failure");
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        auto source = tensor();
        source.type = type;
        auto index = tensor();
        index.type = GGML_TYPE_I32;
        index.ne[1] = 1;
        auto gathered = tensor();
        gathered.op = GGML_OP_GET_ROWS;
        gathered.src[0] = &source;
        gathered.src[1] = &index;
        ctx.tensors[&source] = {1};
        ctx.tensors[&index] = {2};
        for (int64_t count : {1, 2}) {
            index.ne[0] = gathered.ne[1] = count;
            cannge_es::last_reshape_shape.clear();
            check(ggml_cannge_build_tensor(ctx, &gathered), "gather builds");
            const std::vector<int64_t> expected = count == 1 ? std::vector<int64_t>{4} : std::vector<int64_t>{2, 4};
            check(cannge_es::last_reshape_shape == expected, "gather output rank matches downstream ggml axes");
        }
        ctx.tensors.erase(&source);
        ctx.tensors.erase(&index);
        ctx.tensors.erase(&gathered);
    }
    ggml_cannge_plan plan;
    ggml_cannge_plan_io assembly_io;
    auto second = tensor();
    std::strcpy(second.name, "second-output");
    assembly_io.inputs = { &base, &second };
    assembly_io.output_tensors = { &base, &second };
    ggml_cgraph empty_graph{};
    void * built = nullptr;
    cannge_es::fail_output_index = 1;
    check(!ggml_cannge_graph_build(plan, assembly_io, &empty_graph, built, error), "output registration failure aborts assembly");
    check(built == nullptr && cannge_es::build_calls == 0 && cannge_es::output_calls == 2,
          "failed second output does not build a partial graph");
    check(error.find("output[1] second-output: status 17") != std::string::npos,
          "output failure identifies port, tensor and status");
    cannge_es::fail_output_index = -1;
    check(ggml_cannge_graph_build(plan, assembly_io, &empty_graph, built, error), "valid outputs build successfully");
    check(built != nullptr && cannge_es::build_calls == 1, "successful assembly publishes graph");
    delete static_cast<ge::Graph *>(built);
    std::puts("PASS: CANNGE lookup and assembly failure propagation");
}

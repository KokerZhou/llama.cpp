#include "plan.h"
#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(bool result, const char * message) {
    if (!result) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
struct fixture {
    ggml_tensor tensors[4]{};
    ggml_tensor * nodes[2];
    ggml_cgraph graph{};
    fixture() {
        for (auto & t : tensors) {
            t.type = GGML_TYPE_F32;
            t.ne[0] = 4;
            t.ne[1] = t.ne[2] = t.ne[3] = 1;
            t.nb[0] = 4;
            t.nb[1] = t.nb[2] = t.nb[3] = 16;
            t.data = &t;
        }
        tensors[1].op = GGML_OP_SQR;
        tensors[1].src[0] = &tensors[0];
        std::strcpy(tensors[1].name, "intermediate");
        tensors[2].op = GGML_OP_SQR;
        tensors[2].src[0] = &tensors[1];
        nodes[0] = &tensors[1]; nodes[1] = &tensors[2];
        graph.nodes = nodes; graph.n_nodes = 2;
    }
};
int main() {
    fixture f, relocated;
    ggml_cannge_options options;
    std::string err;
    ggml_cannge_plan_io io, other;
    check(ggml_cannge_plan_analyze(&f.graph, io, err, options), "simple analysis");
    check(io.inputs == std::vector<ggml_tensor *>{&f.tensors[0]}, "input order");
    check(io.output_tensors == std::vector<ggml_tensor *>{&f.tensors[2]}, "boundary output");
    check(io.mem_estimate == 48, "memory estimate");
    check(io.input_staged == std::vector<bool>{false}, "dense input");
    check(ggml_cannge_plan_analyze(&relocated.graph, other, err, options), "relocated analysis");
    const auto key = ggml_cannge_plan_signature(&f.graph, io);
    check(key == ggml_cannge_plan_signature(&relocated.graph, other), "address-independent key");
    auto changed = io; changed.options.gemv_f16 = false;
    check(key != ggml_cannge_plan_signature(&f.graph, changed), "option-dependent key");

    options.dump_present = true; options.dump_filter = "intermediate";
    ggml_cannge_plan_io dumped;
    check(ggml_cannge_plan_analyze(&f.graph, dumped, err, options), "dump analysis");
    check(dumped.output_tensors == std::vector<ggml_tensor *>({&f.tensors[1], &f.tensors[2]}), "dump output order");
    check(key == ggml_cannge_plan_signature(&f.graph, io), "original snapshot unchanged");
    auto reordered = dumped;
    std::swap(reordered.output_tensors[0], reordered.output_tensors[1]);
    check(ggml_cannge_plan_signature(&f.graph, dumped) != ggml_cannge_plan_signature(&f.graph, reordered), "port order in key");

    fixture copy;
    copy.tensors[1].op = GGML_OP_DUP;
    ggml_cannge_plan_io distinct, aliased;
    options = {};
    check(ggml_cannge_plan_analyze(&copy.graph, distinct, err, options), "distinct DUP");
    check(distinct.output_tensors == std::vector<ggml_tensor *>({&copy.tensors[1], &copy.tensors[2]}), "DUP output first");
    copy.tensors[1].data = copy.tensors[0].data;
    check(ggml_cannge_plan_analyze(&copy.graph, aliased, err, options), "aliased DUP");
    check(aliased.output_tensors.size() == 1, "alias avoids redundant port");

    fixture scatter;
    scatter.tensors[3].type = GGML_TYPE_I32;
    scatter.tensors[2].op = GGML_OP_SET_ROWS;
    scatter.tensors[2].src[1] = &scatter.tensors[3];
    scatter.tensors[2].src[2] = &scatter.tensors[0];
    ggml_cannge_plan_io scattered;
    check(ggml_cannge_plan_analyze(&scatter.graph, scattered, err, options), "scatter analysis");
    check(scattered.set_rows.size() == 1 && !scattered.set_rows[0].in_graph, "host scatter default");
    check(scattered.set_rows[0].data == &scatter.tensors[1], "scatter data source");
    check(scattered.output_tensors == std::vector<ggml_tensor *>({&scatter.tensors[2], &scatter.tensors[1]}), "scatter materializes producer");

    scatter.tensors[3].ne[0] = 1;
    options.scatter = true;
    ggml_cannge_plan_io in_graph;
    check(ggml_cannge_plan_analyze(&scatter.graph, in_graph, err, options), "in-graph scatter analysis");
    check(in_graph.set_rows[0].in_graph && in_graph.output_tensors.size() == 1, "in-graph scatter avoids producer writeback");
    options.scatter = false;
    options.contiguous_views = true;
    ggml_cannge_plan_io merged;
    check(ggml_cannge_plan_analyze(&scatter.graph, merged, err, options), "merged cache analysis");
    check(merged.set_rows[0].in_graph && merged.output_tensors.size() == 1, "merged cache output deduplicated");

    fixture broken;
    broken.tensors[0].view_src = &broken.tensors[0];
    ggml_cannge_plan_io invalid;
    check(!ggml_cannge_plan_analyze(&broken.graph, invalid, err, options) && !err.empty(), "broken input chain rejected");
    std::puts("PASS: CANNGE plan stages and cache binding regressions");
}

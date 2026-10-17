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

    fixture views;
    auto & cache = views.tensors[0];
    cache.ne[1] = 4; cache.nb[2] = cache.nb[3] = 64;
    auto & transpose = views.tensors[1];
    transpose.op = GGML_OP_TRANSPOSE; transpose.view_src = &cache;
    transpose.ne[0] = 2; transpose.ne[1] = 4;
    transpose.nb[0] = 16; transpose.nb[1] = 4;
    views.graph.nodes = &views.nodes[1]; views.graph.n_nodes = 1;
    ggml_cannge_plan_io rearranged;
    options = {};
    check(ggml_cannge_plan_analyze(&views.graph, rearranged, err, options), "rearranged view analysis");
    check(rearranged.inputs == std::vector<ggml_tensor *>{&cache}, "view binds dense base");
    check(!rearranged.input_staged[0], "base avoids staging");
    auto & cont = views.tensors[3];
    cont.op = GGML_OP_CONT; cont.src[0] = &transpose;
    cont.ne[0] = 2; cont.ne[1] = 4;
    views.tensors[2].src[0] = &cont;
    ggml_cannge_plan_io contiguous;
    check(ggml_cannge_plan_analyze(&views.graph, contiguous, err, options), "CONT view analysis");
    check(contiguous.inputs == std::vector<ggml_tensor *>{&cache}, "CONT binds same base");
    views.tensors[2].src[0] = &transpose;
    transpose.ne[0] = 3; transpose.ne[1] = 1;
    ggml_cannge_plan_io unaligned;
    check(ggml_cannge_plan_analyze(&views.graph, unaligned, err, options), "unaligned view analysis");
    check(unaligned.inputs == std::vector<ggml_tensor *>{&transpose} && unaligned.input_staged[0], "unaligned view keeps staging fallback");

    fixture broken;
    broken.tensors[0].view_src = &broken.tensors[0];
    ggml_cannge_plan_io invalid;
    check(!ggml_cannge_plan_analyze(&broken.graph, invalid, err, options) && !err.empty(), "broken input chain rejected");
    check(invalid.inputs.empty() && invalid.output_tensors.empty() && invalid.mem_estimate == 0, "failed analysis publishes no partial IO");
    std::strcpy(broken.tensors[0].name, "broken_input");
    auto retained = io;
    check(!ggml_cannge_plan_analyze(&broken.graph, retained, err, options), "failed replacement");
    check(err == "invalid input view chain: broken_input", "input diagnostic identifies tensor");
    check(retained.inputs == io.inputs && retained.output_tensors == io.output_tensors &&
          retained.mem_estimate == io.mem_estimate && retained.io_views.size() == io.io_views.size(), "failure preserves previous IO");
    check(key == ggml_cannge_plan_signature(&f.graph, retained), "failure preserves cache key");
    err = "stale error";
    check(ggml_cannge_plan_analyze(&f.graph, io, err, io.options), "reuse with aliased options");
    check(err.empty() && io.inputs.size() == 1 && io.output_tensors.size() == 1 && io.mem_estimate == 48, "reuse replaces ports and accounting");
    check(key == ggml_cannge_plan_signature(&f.graph, io), "reuse preserves key");

    fixture broken_output;
    std::strcpy(broken_output.tensors[2].name, "broken_output");
    broken_output.tensors[2].view_src = &broken_output.tensors[2];
    check(!ggml_cannge_plan_analyze(&broken_output.graph, retained, err, options), "invalid output rejected");
    check(err == "invalid output view chain: broken_output", "output diagnostic identifies tensor");
    check(key == ggml_cannge_plan_signature(&f.graph, retained), "output failure preserves previous analysis");
    std::puts("PASS: CANNGE plan stages and cache binding regressions");
}

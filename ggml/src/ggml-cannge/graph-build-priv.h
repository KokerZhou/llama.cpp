// CANNGE ES context shared by graph assembly, operation mappings and IO builders.

#pragma once

#include "graph-build.h"
#include "plan.h"

#include "ge.h"

#include <ge/es_graph_builder.h>

#include <es_math/es_math_ops.h>
#include <es_nn/es_nn_ops.h>
#include <es_transformer/es_transformer_ops.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace cannge_es = ge::es;

// One step of a pure-rearrangement view chain (defined in graph-build-view.cpp).
bool ggml_cannge_build_relabel(ggml_tensor * step, cannge_es::EsTensorHolder & h, std::string & err);

struct ggml_cannge_build_context;
struct ggml_cannge_set_rows_op;
// CONT of a rearrange view -> in-GE relabel chain + fresh-row splice
bool ggml_cannge_build_cont_chain(ggml_cannge_build_context & ctx, ggml_tensor * t, cannge_es::EsTensorHolder & h);
const ggml_cannge_set_rows_op * ggml_cannge_sr_for_base(ggml_cannge_build_context & ctx, ggml_tensor * base);
bool ggml_cannge_onehot_slot(ggml_cannge_build_context & ctx, cannge_es::EsTensorHolder idxh, int64_t S,
                             cannge_es::EsTensorHolder & mask, ggml_tensor * idx);

struct ggml_cannge_build_context {
    ggml_cannge_plan_io &     io;
    cannge_es::EsGraphBuilder builder;
    std::map<ggml_tensor *, cannge_es::EsTensorHolder> tensors;
    std::string & err;

    // IFA actual_seq_lengths port (int64 [1], backend-owned buffer), created
    // as the last graph input when io.fa_mask is set (note-42)
    cannge_es::EsTensorHolder fa_asl_holder;
    bool                      has_fa_asl = false;

    ggml_cannge_build_context(ggml_cannge_plan_io & io, std::string & err) : io(io), builder("ggml_cannge"), err(err) {
    }

    bool fail(const std::string & msg) {
        err = msg;
        return false;
    }

    bool get(ggml_tensor * t, cannge_es::EsTensorHolder & holder);
};

// External inputs become GE graph inputs in io.inputs order.
bool ggml_cannge_build_inputs(ggml_cannge_build_context & ctx);
bool ggml_cannge_build_view(ggml_cannge_build_context & ctx, ggml_tensor * node);

bool ggml_cannge_build_view_prefix(ggml_cannge_build_context & ctx, const ggml_tensor * tensor,
                                   const ggml_tensor * base, cannge_es::EsTensorHolder & holder);

// CANNGE graph build internals shared between graph-build.cpp (op
// dispatch) and graph-build-io.cpp (inputs/views/relabels).

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

// one step of a pure-rearrangement view chain (defined in graph-build-io.cpp)
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

    bool get(ggml_tensor * t, cannge_es::EsTensorHolder & holder) {
        auto it = tensors.find(t);
        if (it == tensors.end()) {
            // CONT of a rearrange view (merged-attention V path): relabel
            // chain + fresh-row splice (note-50 batch K)
            if (t->op == GGML_OP_CONT && ggml_cannge_build_cont_chain(*this, t, holder)) {
                tensors[t] = holder;
                return true;
            }
            // rearrange-view input substituted by its base (plan.cpp): rebuild
            // the relabel chain in-GE from the registered base holder
            if (t->view_src != nullptr) {
                std::vector<ggml_tensor *> chain;
                ggml_tensor *              base = t;
                while (base->view_src != nullptr) {
                    chain.push_back(base);
                    base = base->view_src;
                }
                auto bit = tensors.find(base);
                if (bit != tensors.end()) {
                    cannge_es::EsTensorHolder h = bit->second;
                    // prefix slice: a folded view may cover only the first
                    // ggml_nelements(t) bytes of a larger base buffer (the KV
                    // cache view spans half the 512-slot context) - take the
                    // linear prefix on GE dim0 before relabeling
                    const int64_t t_elems  = ggml_nelements(t);
                    const int64_t b_elems  = ggml_nelements(base);
                    if (t_elems < b_elems) {
                        std::vector<int64_t> ge = ggml_cannge_ge_shape(base);
                        int64_t              inner = 1;
                        for (size_t k = 1; k < ge.size(); k++) {
                            inner *= ge[k];
                        }
                        if (inner <= 0 || t_elems % inner != 0) {
                            return fail("view prefix does not align to base " + std::string(t->name));
                        }
                        std::vector<int64_t> off(ge.size(), 0);
                        std::vector<int64_t> sz = ge;
                        sz[0]                   = t_elems / inner;
                        h                       = cannge_es::Slice(h, off, sz);
                    }
                    for (auto rit = chain.rbegin(); rit != chain.rend(); ++rit) {
                        if (!ggml_cannge_build_relabel(*rit, h, err)) {
                            return false;
                        }
                    }
                    tensors[t] = h;
                    holder     = h;
                    return true;
                }
            }
            if (getenv("GGML_CANNGE_DEBUG_BUILD")) {
                fprintf(stderr,
                        "[DBG-BUILD] no ES tensor for %s op=%s type=%d ne=%lld,%lld,%lld,%lld nb=%lld,%lld,%lld,%lld\n",
                        t->name, ggml_op_name(t->op), (int) t->type, (long long) t->ne[0], (long long) t->ne[1],
                        (long long) t->ne[2], (long long) t->ne[3], (long long) t->nb[0], (long long) t->nb[1],
                        (long long) t->nb[2], (long long) t->nb[3]);
            }
            return fail("no ES tensor for " + std::string(t->name));
        }
        holder = it->second;
        return true;
    }
};

// external input tensors become GE graph inputs, in io.inputs order so the

bool ggml_cannge_build_inputs(ggml_cannge_build_context & ctx);
bool ggml_cannge_build_view(ggml_cannge_build_context & ctx, ggml_tensor * node);
bool ggml_cannge_build_matmul(ggml_cannge_build_context & ctx, ggml_tensor * node);
bool ggml_cannge_build_attention(ggml_cannge_build_context & ctx, ggml_tensor * node);

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
#include <typeinfo>
#include <vector>

namespace cannge_es = ge::es;

// One step of a pure-rearrangement view chain (defined in graph-build-view.cpp).
bool ggml_cannge_build_relabel(ggml_tensor * step, cannge_es::EsTensorHolder & h, std::string & err);

struct ggml_cannge_build_context;
struct ggml_cannge_set_rows_op;
enum class ggml_cannge_build_result {
    not_applicable,
    success,
    failure,
};

// CONT of a rearrange view -> in-GE relabel chain + fresh-row splice
ggml_cannge_build_result ggml_cannge_build_cont_chain(ggml_cannge_build_context & ctx, ggml_tensor * t, cannge_es::EsTensorHolder & h);
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

    // dedup constants: per-node builders re-emit identical axes/scalars/tables;
    // keep one Const op per distinct value in the compiled graph
    std::map<std::string, cannge_es::EsTensorHolder> consts;

    template <typename T>
    cannge_es::EsTensorHolder const_cached(const std::vector<T> & value, const std::vector<int64_t> & dims) {
        std::string key = "c";
        key += typeid(T).name();
        key.append(reinterpret_cast<const char *>(dims.data()), dims.size() * sizeof(int64_t));
        key.append(reinterpret_cast<const char *>(value.data()), value.size() * sizeof(T));
        auto [it, inserted] = consts.try_emplace(std::move(key));
        if (inserted) {
            it->second = builder.CreateConst(value, dims);
        }
        return it->second;
    }

    template <typename T>
    cannge_es::EsTensorHolder scalar_cached(T value) {
        std::string key = "s";
        key += typeid(T).name();
        key.append(reinterpret_cast<const char *>(&value), sizeof(value));
        auto [it, inserted] = consts.try_emplace(std::move(key));
        if (inserted) {
            it->second = builder.CreateScalar(value);
        }
        return it->second;
    }

    template <typename T>
    cannge_es::EsTensorHolder vector_cached(const std::vector<T> & value) {
        std::string key = "v";
        key += typeid(T).name();
        key.append(reinterpret_cast<const char *>(value.data()), value.size() * sizeof(T));
        auto [it, inserted] = consts.try_emplace(std::move(key));
        if (inserted) {
            it->second = builder.CreateVector(value);
        }
        return it->second;
    }

    // Optional lookup preserves errors for absent tensors, but reports reconstruction failures.
    ggml_cannge_build_result try_get(ggml_tensor * t, cannge_es::EsTensorHolder & holder);
    bool get(ggml_tensor * t, cannge_es::EsTensorHolder & holder);
};

// External inputs become GE graph inputs in io.inputs order.
bool ggml_cannge_build_inputs(ggml_cannge_build_context & ctx);
bool ggml_cannge_build_view(ggml_cannge_build_context & ctx, ggml_tensor * node);

bool ggml_cannge_build_view_prefix(ggml_cannge_build_context & ctx, const ggml_tensor * tensor,
                                   const ggml_tensor * base, cannge_es::EsTensorHolder & holder);

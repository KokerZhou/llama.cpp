#include "plan.h"
#include "ggml-impl.h"

#include <unordered_map>

std::string ggml_cannge_plan_signature(ggml_cgraph * cgraph, const ggml_cannge_plan_io & io) {
    std::string sig;

    // tensor ids by first appearance: srcs of a node before the node itself
    std::unordered_map<ggml_tensor *, uint32_t> ids;
    std::vector<ggml_tensor *> tensors;
    auto tensor_id = [&ids, &tensors](ggml_tensor * t) -> uint32_t {
        auto it = ids.find(t);
        if (it != ids.end()) {
            return it->second;
        }
        uint32_t id = static_cast<uint32_t>(ids.size());
        ids[t] = id;
        tensors.push_back(t);
        return id;
    };
    auto append_i32 = [&sig](int32_t v) { sig.append(reinterpret_cast<const char *>(&v), sizeof(v)); };
    auto append_u32 = [&sig](uint32_t v) { sig.append(reinterpret_cast<const char *>(&v), sizeof(v)); };
    auto append_i64 = [&sig](int64_t v) { sig.append(reinterpret_cast<const char *>(&v), sizeof(v)); };
    auto append_u64 = [&sig](uint64_t v) { sig.append(reinterpret_cast<const char *>(&v), sizeof(v)); };
    auto append_meta = [&sig, append_i32, append_i64, append_u64](const ggml_tensor * t) {
        append_i32(static_cast<int32_t>(t->type));
        append_u64(t->view_offs);
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            append_i64(t->ne[i]);
        }
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            append_u64(t->nb[i]);
        }
    };

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        append_i32(static_cast<int32_t>(node->op));
        append_i32(node->flags);
        append_meta(node);
        sig.append(reinterpret_cast<const char *>(node->op_params), sizeof(node->op_params));

        int n_srcs = 0;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                n_srcs++;
            }
        }
        append_i32(n_srcs);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            ggml_tensor * src = node->src[j];
            if (src == nullptr) {
                continue;
            }
            append_i32(j);
            append_u32(tensor_id(src));
            append_meta(src);
        }
        append_u32(tensor_id(node));
    }

    // Port order is part of the compiled graph's binding contract.
    // Walk-assigned ids keep the key stable when tensor addresses change.
    auto append_ports = [&](const std::vector<ggml_tensor *> & ports) {
        append_u64(ports.size());
        for (ggml_tensor * t : ports) {
            append_u32(tensor_id(t));
            append_meta(t);
        }
    };
    append_ports(io.inputs);
    append_ports(io.output_tensors);
    append_u64(io.set_rows.size());
    for (const ggml_cannge_set_rows_op & sr : io.set_rows) {
        append_u32(tensor_id(sr.dst));
        append_u32(tensor_id(sr.data));
        append_u32(tensor_id(sr.idx));
        append_i32(sr.in_graph);
    }

    auto append_reference = [&](ggml_tensor * tensor) {
        append_i32(tensor != nullptr);
        if (tensor != nullptr) {
            append_u32(tensor_id(tensor));
        }
    };

    append_u64(io.input_base.size());
    for (size_t i = 0; i < io.input_base.size(); i++) {
        append_reference(io.input_base[i]);
        append_u64(io.input_base_offset[i]);
    }

    // Builders replay external view and CONT chains. Record those dependencies
    // without traversing the unrelated producer graphs of staged inputs.
    for (size_t i = 0; i < tensors.size(); i++) {
        ggml_tensor * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }
        if (tensor->view_src != nullptr) {
            tensor_id(tensor->view_src);
        }
        if ((tensor->view_src != nullptr || tensor->op == GGML_OP_CONT) && tensor->src[0] != nullptr) {
            tensor_id(tensor->src[0]);
        }
    }
    append_u64(tensors.size());
    for (ggml_tensor * tensor : tensors) {
        append_i32(tensor != nullptr);
        if (tensor == nullptr) {
            continue;
        }
        append_meta(tensor);
        append_i32(static_cast<int32_t>(tensor->op));
        sig.append(reinterpret_cast<const char *>(tensor->op_params), sizeof(tensor->op_params));
        append_reference(tensor->view_src);
        append_reference(tensor->view_src != nullptr || tensor->op == GGML_OP_CONT ? tensor->src[0] : nullptr);
    }

    sig.append(io.options.signature());

    return sig;
}

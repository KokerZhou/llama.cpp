#include "view.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

static bool ggml_cannge_view_chain_acyclic(const ggml_tensor * t) {
    // Detect cycles without allocating on the per-compute analysis path.
    const ggml_tensor * slow = t;
    const ggml_tensor * fast = t;
    while (fast != nullptr && fast->view_src != nullptr) {
        slow = slow->view_src;
        fast = fast->view_src->view_src;
        if (slow == fast) {
            return false;
        }
    }

    return true;
}

// walk the view_src chain down to the root tensor
bool ggml_cannge_resolve_view(ggml_tensor * t, ggml_cannge_view_info & info) {
    info = {};
    if (t == nullptr) {
        return false;
    }

    if (!ggml_cannge_view_chain_acyclic(t)) {
        return false;
    }

    ggml_tensor * base = t;
    while (ggml_is_view(base)) {
        if (base->view_offs > std::numeric_limits<size_t>::max() - info.offset) {
            return false;
        }
        info.offset += base->view_offs;
        base = base->view_src;
        if (base == nullptr) {
            return false;
        }
    }
    info.base = base;

    // nb of the view itself is already expressed relative to its data pointer
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        info.nb[i] = t->nb[i];
    }

    info.is_dense = info.nb[0] == ggml_element_size(t);
    for (int i = 1; i < GGML_MAX_DIMS && info.is_dense; i++) {
        info.is_dense = info.nb[i] == info.nb[i - 1] * (size_t) t->ne[i - 1];
    }
    return true;
}

// every step of the view_src chain only relabels bytes (no element transform):
// such a view of a fresh input buffer already holds the correct content for
// consumers reading through the view strides, so GE needs no output port
bool ggml_cannge_is_rearrange_view(const ggml_tensor * t) {
    if (t == nullptr || !ggml_cannge_view_chain_acyclic(t)) {
        return false;
    }
    for (const ggml_tensor * cur = t; cur != nullptr && cur->view_src != nullptr; cur = cur->view_src) {
        if (cur->op != GGML_OP_RESHAPE && cur->op != GGML_OP_PERMUTE && cur->op != GGML_OP_TRANSPOSE &&
            cur->op != GGML_OP_VIEW && cur->op != GGML_OP_NONE) {
            return false;
        }
    }
    return true;
}

// resolve a view to a registered base tensor, mirroring build_view's chain walk
// and tiered candidate search
ggml_tensor * ggml_cannge_resolve_view_base(ggml_tensor * node, const std::set<ggml_tensor *> & registered, size_t & offset) {
    offset = 0;
    if (node == nullptr || !ggml_is_view(node) || !ggml_cannge_view_chain_acyclic(node)) {
        return nullptr;
    }

    // chain walk: follow view_src while it is registered (has_holder)
    ggml_tensor * base = node;
    while (ggml_is_view(base) && base->view_src != nullptr) {
        if (registered.find(base->view_src) == registered.end()) {
            break;
        }
        if (base->view_offs > std::numeric_limits<size_t>::max() - offset) {
            offset = 0;
            return nullptr;
        }
        offset += base->view_offs;
        base    = base->view_src;
    }
    if (registered.count(base)) {
        return base;
    }

    if (node->data == nullptr) {
        return nullptr;
    }

    const int64_t esize   = ggml_element_size(node);
    const int     node_nd = ggml_n_dims(node);
    if (esize <= 0) {
        return nullptr;
    }
    size_t node_span = esize;
    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        if (node->ne[d] <= 0 || (uint64_t) node->ne[d] > std::numeric_limits<size_t>::max() / node_span) {
            return nullptr;
        }
        node_span *= (size_t) node->ne[d];
    }

    auto fits = [&](const ggml_tensor * cand) -> bool {
        const uintptr_t lo = reinterpret_cast<uintptr_t>(cand->data);
        const uintptr_t address = reinterpret_cast<uintptr_t>(node->data);
        const size_t span = ggml_nbytes(cand);
        if (cand->data == nullptr || address < lo || address - lo >= span) {
            return false;
        }
        // whole-buffer reinterpretation: same address, same byte span
        if (address == lo && node_span == span) {
            return true;
        }
        const uintptr_t byte_offset = address - lo;
        if (byte_offset % esize != 0 || byte_offset / esize > (uint64_t) std::numeric_limits<int64_t>::max()) {
            return false;
        }
        int64_t rem = (int64_t) (byte_offset / esize);
        const int cand_nd = ggml_n_dims(cand);
        for (int d = 0; d < cand_nd; d++) {
            const int64_t cand_ne = cand->ne[d];
            if (cand_ne <= 0) {
                return false;
            }
            const int64_t off = rem % cand_ne;
            rem /= cand_ne;
            if (d < node_nd && node->ne[d] > cand_ne - off) {
                return false;
            }
        }
        for (int d = cand_nd; d < node_nd; d++) {
            if (node->ne[d] != 1) {
                return false;
            }
        }
        return rem == 0;
    };

    auto rank_of = [](const ggml_tensor * t) { return ggml_n_dims(t); };

    // collect all fits and order them deterministically: tightest span
    // first, then rank penalty, name, address as the last resort. A pointer-
    // ordered tie used to pick an arbitrary base when allocator reuse made
    // several candidates contain the view (rogue shapes, note-46)
    std::vector<const ggml_tensor *> fits_vec;
    auto collect = [&](bool name_match_only) {
        fits_vec.clear();
        for (ggml_tensor * cand : registered) {
            if (name_match_only && (node->view_src == nullptr || strcmp(cand->name, node->view_src->name) != 0)) {
                continue;
            }
            if (fits(cand)) {
                fits_vec.push_back(cand);
            }
        }
        std::sort(fits_vec.begin(), fits_vec.end(), [&](const ggml_tensor * a, const ggml_tensor * b) {
            const size_t sa = ggml_nbytes(a);
            const size_t sb = ggml_nbytes(b);
            if (sa != sb) return sa < sb;
            const int pa = rank_of(a) == node_nd ? 0 : 1;
            const int pb = rank_of(b) == node_nd ? 0 : 1;
            if (pa != pb) return pa < pb;
            const int c = strcmp(a->name, b->name);
            if (c != 0) return c < 0;
            return std::less<const ggml_tensor *>{}(a, b);
        });
    };

    const ggml_tensor * best = nullptr;
    // Tier 1: registered tensor whose name matches the immediate view_src
    if (node->view_src != nullptr) {
        collect(true);
        if (!fits_vec.empty()) {
            best = fits_vec.front();
        }
    }

    // Tier 2: any registered tensor whose buffer contains node->data
    if (best == nullptr) {
        collect(false);
        if (!fits_vec.empty()) {
            if (fits_vec.size() > 1 && getenv("GGML_CANNGE_DEBUG_COMPILE")) {
                fprintf(stderr, "[DBG-VIEW] view %s: %zu candidate bases, picking %s (span %zu)\n", node->name,
                        fits_vec.size(), fits_vec.front()->name, ggml_nbytes(fits_vec.front()));
            }
            best = fits_vec.front();
        }
    }

    if (best != nullptr) {
        base   = (ggml_tensor *) best;
        offset = reinterpret_cast<uintptr_t>(node->data) - reinterpret_cast<uintptr_t>(best->data);
        return base;
    }

    return nullptr;
}

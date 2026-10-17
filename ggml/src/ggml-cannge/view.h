#pragma once

#include "ggml.h"
#include <set>

// resolved view chain of one IO tensor: the base it aliases, the byte offset
// of its data inside the base, and its effective strides (already valid
// relative to its own data pointer)
struct ggml_cannge_view_info {
    ggml_tensor * base = nullptr;
    size_t        offset = 0;
    size_t        nb[GGML_MAX_DIMS];
    bool          is_dense = false;
};

// walk the view_src chain down to the root tensor (shared with the build step)
bool ggml_cannge_resolve_view(ggml_tensor * t, ggml_cannge_view_info & info);

// resolve a view to a registered base tensor and the byte offset of its data
// inside that base; returns nullptr if the view cannot be resolved
ggml_tensor * ggml_cannge_resolve_view_base(ggml_tensor * node, const std::set<ggml_tensor *> & registered, size_t & offset);

bool ggml_cannge_is_rearrange_view(const ggml_tensor * tensor);

// Return the outer extent of a positive, aligned prefix no larger than its base.
bool ggml_cannge_view_prefix_extent(const ggml_tensor * tensor, const ggml_tensor * base, int64_t & outer);

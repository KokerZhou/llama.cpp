#include "view.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

static void check(bool result, const char * description) {
    if (!result) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}
static ggml_tensor tensor(void * data, int64_t width, int64_t rows = 1) {
    ggml_tensor t{};
    t.type = GGML_TYPE_F32;
    t.data = data;
    t.ne[0] = width;
    t.ne[1] = rows;
    t.ne[2] = t.ne[3] = 1;
    t.nb[0] = 4;
    t.nb[1] = width * 4;
    t.nb[2] = t.nb[3] = width * rows * 4;
    return t;
}
int main() {
    char storage[256]{};
    auto base = tensor(storage, 4, 8);
    auto middle = tensor(storage + 16, 4, 4);
    auto leaf = tensor(storage + 32, 4, 2);
    middle.view_src = &base; middle.view_offs = 16; middle.op = GGML_OP_VIEW;
    leaf.view_src = &middle; leaf.view_offs = 16; leaf.op = GGML_OP_RESHAPE;
    ggml_cannge_view_info info;
    size_t offset = 0;
    check(ggml_cannge_resolve_view(&leaf, info) && info.base == &base && info.offset == 32, "nested offsets");
    check(info.is_dense && info.nb[1] == 16, "view strides");
    check(ggml_cannge_is_rearrange_view(&leaf), "metadata chain");
    middle.op = GGML_OP_ADD;
    check(!ggml_cannge_is_rearrange_view(&leaf), "compute step rejected");
    middle.op = GGML_OP_VIEW;
    check(ggml_cannge_resolve_view_base(&leaf, {&middle, &base}, offset) == &base && offset == 32, "highest registered base");
    check(ggml_cannge_resolve_view_base(&leaf, {&middle}, offset) == &middle && offset == 16, "registered intermediate");
    middle.view_offs = std::numeric_limits<size_t>::max();
    check(!ggml_cannge_resolve_view(&leaf, info), "chain offset overflow");
    check(ggml_cannge_resolve_view_base(&leaf, {&middle, &base}, offset) == nullptr && offset == 0, "registered offset overflow");
    middle.view_offs = 16;
    base.view_src = &leaf;
    check(!ggml_cannge_resolve_view(&leaf, info), "cycle in resolution");
    check(!ggml_cannge_is_rearrange_view(&leaf), "cycle in predicate");
    check(ggml_cannge_resolve_view_base(&leaf, {&middle, &base}, offset) == nullptr, "cycle in base search");
    base.view_src = nullptr;
    check(!ggml_cannge_is_rearrange_view(nullptr), "null predicate");
    check(!ggml_cannge_resolve_view(nullptr, info), "null resolution");

    auto missing = tensor(storage, 4, 8);
    auto view = tensor(storage + 32, 4, 2);
    view.view_src = &missing;
    check(ggml_cannge_resolve_view_base(&view, {&base}, offset) == &base && offset == 32, "address fallback");
    auto tight = tensor(storage + 16, 4, 4);
    std::strcpy(missing.name, "source");
    std::strcpy(base.name, "other");
    std::strcpy(tight.name, "tight");
    check(ggml_cannge_resolve_view_base(&view, {&base, &tight}, offset) == &tight && offset == 16, "tightest candidate");
    std::strcpy(base.name, "source");
    check(ggml_cannge_resolve_view_base(&view, {&base, &tight}, offset) == &base && offset == 32, "name tier wins");
    view.data = storage + 33;
    check(ggml_cannge_resolve_view_base(&view, {&base}, offset) == nullptr, "misaligned address");
    view.data = storage + 128;
    check(ggml_cannge_resolve_view_base(&view, {&base}, offset) == nullptr, "end-exclusive span");
    view.data = storage;
    view.ne[0] = std::numeric_limits<int64_t>::max();
    check(ggml_cannge_resolve_view_base(&view, {&base}, offset) == nullptr, "footprint overflow");
    view.ne[0] = 0;
    check(ggml_cannge_resolve_view_base(&view, {&base}, offset) == nullptr, "zero footprint");
    int64_t outer = -1;
    auto prefix = tensor(storage, 4, 2);
    check(ggml_cannge_view_prefix_extent(&prefix, &base, outer) && outer == 2, "aligned prefix extent");
    check(ggml_cannge_view_prefix_extent(&base, &base, outer) && outer == 8, "full prefix extent");
    prefix.ne[1] = 9;
    check(!ggml_cannge_view_prefix_extent(&prefix, &base, outer) && outer == 0, "oversized prefix rejected");
    prefix.ne[0] = 3; prefix.ne[1] = 1;
    check(!ggml_cannge_view_prefix_extent(&prefix, &base, outer), "unaligned prefix rejected");
    prefix.ne[0] = 0;
    check(!ggml_cannge_view_prefix_extent(&prefix, &base, outer), "empty prefix rejected");
    prefix.ne[0] = std::numeric_limits<int64_t>::max(); prefix.ne[1] = 2;
    check(!ggml_cannge_view_prefix_extent(&prefix, &base, outer), "prefix element overflow");
    check(!ggml_cannge_view_prefix_extent(&base, &prefix, outer), "base element overflow");
    check(!ggml_cannge_view_prefix_extent(nullptr, &base, outer), "missing prefix");
    auto vector = tensor(storage, 8); prefix = tensor(storage, 3);
    check(ggml_cannge_view_prefix_extent(&prefix, &vector, outer) && outer == 3, "one-dimensional prefix");
    std::puts("PASS: CANNGE view metadata regressions");
}

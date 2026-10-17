/*
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "ggml-cannge-impl.h"

#include <mutex>

ggml_backend_cannge_context::ggml_backend_cannge_context(int32_t device) : device(device), name(GGML_CANNGE_NAME + std::to_string(device)) {
    if (!ggml_cannge_check(aclrtSetDevice(device), "aclrtSetDevice", __func__)) {
        return;
    }
    initialized = ggml_cannge_check(aclrtCreateStream(&stream), "aclrtCreateStream", __func__) && stream != nullptr;
}

ggml_backend_cannge_context::~ggml_backend_cannge_context() = default;

bool ggml_backend_cannge_context::evict_oldest_plan() {
    auto oldest = plans.end();
    for (auto it = plans.begin(); it != plans.end(); ++it) {
        if (oldest == plans.end() || it->second->seq < oldest->second->seq) {
            oldest = it;
        }
    }
    if (oldest != plans.end()) {
        // Keep the graph and its buffers until queued work and graph removal both complete.
        if (stream != nullptr &&
            !ggml_cannge_check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream", __func__)) {
            return false;
        }
        if (ge != nullptr && oldest->second->graph_id != 0 &&
            !ge->remove(oldest->second->graph_id)) {
            return false;
        }
        oldest->second->graph_id = 0;
        if (!oldest->second->release_buffers()) {
            return false;
        }
        plans.erase(oldest);
    }
    return true;
}

bool ggml_cannge_plan::release_buffers() {
    state = FAILED;
    bool released = true;
    auto release = [&](void *& pointer, bool host = false) {
        if (pointer == nullptr) {
            return;
        }
        const aclError error = host ? aclrtFreeHost(pointer) : aclrtFree(pointer);
        if (ggml_cannge_check(error, host ? "aclrtFreeHost" : "aclrtFree", __func__)) {
            pointer = nullptr;
        } else {
            released = false;
        }
    };
    for (void *& pointer : staging) {
        release(pointer);
    }
    for (void *& pointer : input_staging) {
        release(pointer);
    }
    release(in_arena);
    release(in_arena_host, true);
    release(fa_asl_dev);
    return released;
}

ggml_cannge_plan::~ggml_cannge_plan() {
    release_buffers();
}

void ggml_cannge_release_context(ggml_backend_cannge_context * ctx) {
    static std::mutex mutex;
    // Raw links deliberately retain failed contexts at process exit rather than running unsafe destructors.
    static ggml_backend_cannge_context * retired = nullptr;
    std::lock_guard<std::mutex> lock(mutex);
    if (ctx != nullptr) {
        ctx->next_retired = retired;
        retired = ctx;
    }
    auto ** link = &retired;
    while (*link != nullptr) {
        auto * candidate = *link;
        if (candidate->stream != nullptr &&
            (!ggml_cannge_check(aclrtSetDevice(candidate->device), "select retired context device", __func__) ||
             !ggml_cannge_check(aclrtSynchronizeStream(candidate->stream), "drain retired context", __func__))) {
            link = &candidate->next_retired;
            continue;
        }
        // Completion is established; session and buffers no longer need the stream.
        candidate->ge.reset();
        bool released = true;
        for (auto & entry : candidate->plans) {
            if (!entry.second->release_buffers()) {
                released = false;
            }
        }
        if (!released) {
            link = &candidate->next_retired;
            continue;
        }
        candidate->plans.clear();
        if (candidate->stream != nullptr) {
            if (!ggml_cannge_check(aclrtDestroyStream(candidate->stream), "destroy retired context stream", __func__)) {
                link = &candidate->next_retired;
                continue;
            }
            candidate->stream = nullptr;
        }
        *link = candidate->next_retired;
        candidate->next_retired = nullptr;
        delete candidate;
    }
}

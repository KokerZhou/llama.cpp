#pragma once

#include "common.h"
#include <cstdio>

// Drain producers before reading either host or device storage. Never print failed snapshots.
bool ggml_cannge_debug_tensor(const char * tag, const ggml_tensor * tensor, aclrtStream stream, FILE * output = stderr);

#include "ggml.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

void ggml_abort(const char * file, int line, const char * format, ...) {
    std::fprintf(stderr, "%s:%d: ", file, line);
    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::abort();
}

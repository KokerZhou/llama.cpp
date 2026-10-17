#pragma once
#include <cstdint>

using aclError = int;
using aclrtStream = void *;
constexpr aclError ACL_SUCCESS = 0;

aclError aclrtSetDevice(int32_t device);
aclError aclrtCreateStream(aclrtStream * stream);
aclError aclrtSynchronizeStream(aclrtStream stream);
aclError aclrtDestroyStream(aclrtStream stream);
aclError aclrtFree(void * ptr);
aclError aclrtFreeHost(void * ptr);

#include <cstddef>
enum aclrtMemcpyKind { ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST, ACL_MEMCPY_DEVICE_TO_DEVICE };
constexpr int ACL_MEM_MALLOC_HUGE_FIRST = 0;
aclError aclrtMemcpy(void * dst, size_t capacity, const void * src, size_t size, aclrtMemcpyKind kind);
aclError aclrtMemcpyAsync(void * dst, size_t capacity, const void * src, size_t size, aclrtMemcpyKind kind, aclrtStream stream);
aclError aclrtMemcpy2dAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, aclrtMemcpyKind kind, aclrtStream stream);
aclError aclrtMalloc(void ** ptr, size_t size, int policy);
aclError aclrtMallocHost(void ** ptr, size_t size);

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

if ("cannge${CANN_INSTALL_DIR}" STREQUAL "cannge" AND DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(CANN_INSTALL_DIR $ENV{ASCEND_TOOLKIT_HOME})
    message(STATUS "CANNGE: updated CANN_INSTALL_DIR from ASCEND_TOOLKIT_HOME=$ENV{ASCEND_TOOLKIT_HOME}")
endif()

if (CANN_INSTALL_DIR)
    # Only Support Linux.
    if (NOT UNIX)
        message(FATAL_ERROR "CANNGE: CANN toolkit supports unix but not ${CMAKE_SYSTEM_NAME}")
    endif()

    # Supported platforms: x86-64, arm64
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
    else()
        message(FATAL_ERROR "CANNGE: CANN toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    # SOC type: prefer -DSOC_TYPE=..., fall back to npu-smi, do not fail the
    # build when neither is available (compile-only environments have no NPU)
    if (NOT SOC_TYPE)
        execute_process(
            COMMAND bash -c "npu-smi info|awk -F' ' 'NF > 0 && NR==7 {print $3}'"
            OUTPUT_VARIABLE npu_info
            RESULT_VARIABLE npu_result
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if (NOT "${npu_info}" STREQUAL "" AND ${npu_result} EQUAL 0)
            set(SOC_TYPE "Ascend${npu_info}")
        else()
            message(WARNING "CANNGE: SOC type not detected (no npu-smi), continuing without SOC define; specify with -DSOC_TYPE=Ascend310P")
        endif()
    endif()

    if (SOC_TYPE)
        string(TOLOWER ${SOC_TYPE} SOC_VERSION)
        string(REGEX MATCH "[0-9]+[a-zA-Z]" SOC_TYPE_MAJOR_SN "${SOC_VERSION}")
        set(SOC_TYPE_COMPILE_OPTION "ASCEND_${SOC_TYPE_MAJOR_SN}")
        string(TOUPPER ${SOC_TYPE_COMPILE_OPTION} SOC_TYPE_COMPILE_OPTION)
        message(STATUS "CANNGE: SOC_VERSION = ${SOC_VERSION}")
    endif()

    set(CANN_INCLUDE_DIRS
        ${CANN_INSTALL_DIR}/include
        ${CANN_INSTALL_DIR}/include/aclnn
        ${CANN_INSTALL_DIR}/acllib/include
    )

    set(CANN_LINK_DIRS
        ${CANN_INSTALL_DIR}/lib64
        ${CANN_INSTALL_DIR}/lib64/stub
        ${CANN_INSTALL_DIR}/acllib/lib64
        ${CANN_INSTALL_DIR}/acllib/lib64/stub
    )

    # newer run packages keep arch-specific artifacts under <cann>/<arch>-linux/
    foreach(arch x86_64-linux aarch64-linux)
        if (EXISTS ${CANN_INSTALL_DIR}/${arch})
            list(APPEND CANN_INCLUDE_DIRS ${CANN_INSTALL_DIR}/${arch}/include)
            list(APPEND CANN_LINK_DIRS ${CANN_INSTALL_DIR}/${arch}/lib64 ${CANN_INSTALL_DIR}/${arch}/lib64/stub)
        endif()
    endforeach()

    # ES graph authoring frontend (ge::es::EsGraphBuilder + es_* op headers);
    # the graph build step is ES based, so ES is required for this backend
    set(GGML_CANNGE_HAS_ES OFF)
    find_file(ES_C_GRAPH_BUILDER_H es_graph_builder.h PATHS ${CANN_INCLUDE_DIRS} PATH_SUFFIXES ge)
    find_file(ES_MATH_OPS_H es_math_ops.h PATHS ${CANN_INCLUDE_DIRS} PATH_SUFFIXES es/es_math)
    if (ES_C_GRAPH_BUILDER_H AND ES_MATH_OPS_H)
        set(GGML_CANNGE_HAS_ES ON)
        list(APPEND CANN_INCLUDE_DIRS
             ${CANN_INSTALL_DIR}/include/ge
             ${CANN_INSTALL_DIR}/include/es
        )
        message(STATUS "CANNGE: ES graph builder detected (${ES_C_GRAPH_BUILDER_H})")
    else()
        message(FATAL_ERROR "CANNGE: ES headers not found under ${CANN_INSTALL_DIR}/include, ES is required")
    endif()

    ggml_add_backend_library(ggml-cannge
                             ggml-cannge.cpp
                             plan.cpp
                             ge.cpp
                             graph-build.cpp
                            )

    target_include_directories(ggml-cannge PRIVATE ${CANN_INCLUDE_DIRS})
    target_link_directories(ggml-cannge PRIVATE ${CANN_LINK_DIRS})

    # ge_runner: GE session API, graph: GE IR objects, ES authoring libs
    target_link_libraries(ggml-cannge PRIVATE ascendcl ge_runner graph
                          es_math es_nn eager_style_graph_builder_base)

    if (SOC_TYPE)
        target_compile_definitions(ggml-cannge PRIVATE ${SOC_TYPE_COMPILE_OPTION})
    endif()

    if (GGML_CANNGE_HAS_ES)
        target_compile_definitions(ggml-cannge PRIVATE GGML_CANNGE_HAS_ES)
    endif()

    # workarounds from the JittorInfer ES port: the ES-generated package pulls
    # in a ge_error_codes.h without GE_FUNC_VISIBILITY, and CANN libs are
    # built with the old C++ ABI
    target_compile_definitions(ggml-cannge PRIVATE GE_FUNC_VISIBILITY=)
    # CANN libs are built with the old C++ ABI, compile the backend the same
    target_compile_definitions(ggml-cannge PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
else()
    message(FATAL_ERROR "CANNGE: CANN_INSTALL_DIR not set, please set env ASCEND_TOOLKIT_HOME or -DCANN_INSTALL_DIR")
endif()

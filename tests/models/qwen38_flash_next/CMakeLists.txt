# Pre-migration serialization check needs neither HIP nor model weights:
# it compiles continuation_identity.cpp itself for its own fixture path and
# a HIP-free compile, instead of linking gufo_serving.
add_executable(qwen38_flash_next_continuation_identity_test continuation_identity_test.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen38_flash_next/continuation_identity.cpp)
target_include_directories(qwen38_flash_next_continuation_identity_test PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_definitions(qwen38_flash_next_continuation_identity_test PRIVATE
  GUFO_IDENTITY_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/continuation_identity")
add_test(NAME qwen38_flash_next_continuation_identity_test COMMAND qwen38_flash_next_continuation_identity_test)
set_tests_properties(qwen38_flash_next_continuation_identity_test PROPERTIES LABELS "cpu;qwen38_flash_next;cache;correctness")

# Small operator checks are the default. Full-model qualification and
# diagnostic probes are built explicitly so routine iterations stay focused.
add_custom_target(qwen38_flash_next_tests)

function(add_qwen38_flash_next_test name library labels)
  set(target qwen38_flash_next_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME qwen38_flash_next.${name} COMMAND ${target})
  set_tests_properties(qwen38_flash_next.${name} PROPERTIES
    LABELS "qwen38_flash_next;${labels}")
  add_dependencies(qwen38_flash_next_tests ${target})
endfunction()

add_qwen38_flash_next_test(runner_options gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(probe gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(config gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(continuation_layout gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(ngram gufo_qwen38_flash_next "cpu")
add_qwen38_flash_next_test(mtp_sampling gufo_core "cpu")
add_qwen38_flash_next_test(rope_scaling gufo_qwen38_flash_next "cpu")
# The HF-oracle/off-bit-identity checks link the scalar model oracle, so
# (like ngram) this stays out of cmake/Checks.cmake's hosted check-pr set.
add_qwen38_flash_next_test(rope_scaling_oracle
  gufo_qwen38_flash_next_reference "cpu")
# Point 85: I4R128 rotated-expert loader contract and the CPU-oracle parity
# of the rotated MoE path against a materialized dense reference.
add_qwen38_flash_next_test(i4r_loader gufo_qwen38_flash_next_reference "cpu")
add_qwen38_flash_next_test(i4r_moe_parity gufo_qwen38_flash_next_reference "cpu")
add_qwen38_flash_next_test(prompt_lookup gufo_core "cpu")

add_executable(qwen38_flash_next_reference_probe EXCLUDE_FROM_ALL reference_probe.cpp)
target_link_libraries(qwen38_flash_next_reference_probe PRIVATE
  gufo_qwen38_flash_next_reference)
set_source_files_properties(reference_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  add_library(qwen38_flash_next_test_hip INTERFACE)
  target_link_libraries(qwen38_flash_next_test_hip INTERFACE
    gufo_qwen38_flash_next_hip gufo_qwen38_flash_next_mmq hip::host)
  target_compile_definitions(qwen38_flash_next_test_hip INTERFACE ENGINE_ENABLE_HIP=1)
  foreach(name IN ITEMS mtp_ops blaslt_ops hc_mix_ops hc_route_ops gdn_ops
      attention_ops moe_epilogue_ops moe_ids_ops projection_ops routed_wmma_ops
      select_ops snapshot_copy verification_sampling wide_verification)
    add_qwen38_flash_next_test(${name} qwen38_flash_next_test_hip "hip;gpu;gfx1151")
  endforeach()
  # Point 85 step 2: the real-executor rotated-path parity driver. This is
  # the compile+link proof requested in review-v5 blocker 2 -- a real
  # executable target with the full dependency closure (executor + device
  # model + mmq kernels + CPU reference oracle + gufo core) and HIP 20, not
  # a `-c` object-only compile. It is EXCLUDE_FROM_ALL so routine builds
  # neither compile nor run it; select it explicitly with
  #   cmake --build --preset <cfg> --target i4r_executor_parity_test
  # CTest registration is a separate, explicit opt-in (review-v6): labels
  # carry no exclusion semantics -- gpu-full has no filter and gpu-fast
  # excludes only slow|external-model -- so an unconditional add_test would
  # let ordinary GPU presets select this individually-approved
  # qualification (or fail Not Run on the absent executable). The build
  # target stays selectable in every configuration; the test is registered
  # only when GUFO_I4R_PARITY_CTEST is ON. The driver still self-skips
  # (return 77) with no device; SKIP_RETURN_CODE only maps a reported
  # no-device exit to skip, it does not gate selection.
  option(GUFO_I4R_PARITY_CTEST
    "Register qwen38_flash_next.i4r_executor_parity with CTest. Default OFF: the individually approved qualification worker builds the target and runs the exact binary directly; enable only in a dedicated qualification configuration" OFF)
  add_executable(i4r_executor_parity_test EXCLUDE_FROM_ALL
    i4r_executor_parity_test.hip)
  set_target_properties(i4r_executor_parity_test PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON)
  target_link_libraries(i4r_executor_parity_test PRIVATE
    qwen38_flash_next_test_hip
    gufo_qwen38_flash_next_reference
    gufo_core)
  # Prefill dense GEMM: the engine's f16 kernel against hipBLASLt at the v2
  # prefill shapes. Manual only; needs a GPU.
  add_executable(prefill_gemm_bench EXCLUDE_FROM_ALL prefill_gemm_bench.hip)
  set_target_properties(prefill_gemm_bench PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON)
  target_link_libraries(prefill_gemm_bench PRIVATE qwen38_flash_next_test_hip gufo_core
    roc::hipblaslt hip::host)
  # Batched i4r MoE decode: microbenchmark plus real-geometry cross-check
  # against the unbatched dispatch. Manual only; needs a GPU.
  add_executable(i4r_fused_bench EXCLUDE_FROM_ALL i4r_fused_bench.hip)
  set_target_properties(i4r_fused_bench PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON)
  target_link_libraries(i4r_fused_bench PRIVATE gufo_core hip::host)
  if (GUFO_I4R_PARITY_CTEST)
    add_test(NAME qwen38_flash_next.i4r_executor_parity
      COMMAND i4r_executor_parity_test)
    set_tests_properties(qwen38_flash_next.i4r_executor_parity PROPERTIES
      LABELS "qwen38_flash_next;hip;gpu;gfx1151;i4r;manual"
      SKIP_RETURN_CODE 77
      RUN_SERIAL TRUE)
  endif()
  # Reuse the synthetic model fixture, with short prompts around a reduced
  # sparse budget. This exercises actual Executor capture/replay without GGUFs.
  add_test(NAME qwen38_flash_next.decode_graph
    COMMAND qwen38_flash_next_wide_verification_test --decode-graph-only)
  set_tests_properties(qwen38_flash_next.decode_graph PROPERTIES
    LABELS "qwen38_flash_next;hip;gpu;gfx1151")
  # The rope phase and RMS checks compare against the scalar CPU oracle.
  target_link_libraries(qwen38_flash_next_attention_ops_test PRIVATE
    gufo_qwen38_flash_next_reference)

  add_executable(qwen38_flash_next_host_pressure_probe EXCLUDE_FROM_ALL
    host_pressure_probe.cpp)
  target_link_libraries(qwen38_flash_next_host_pressure_probe PRIVATE
    gufo_serving qwen38_flash_next_test_hip)
  add_executable(qwen38_flash_next_session_test EXCLUDE_FROM_ALL session_test.cpp)
  target_link_libraries(qwen38_flash_next_session_test PRIVATE gufo_serving
    qwen38_flash_next_test_hip)
  add_executable(qwen38_flash_next_snapshot_test EXCLUDE_FROM_ALL snapshot_test.cpp
    device_headroom_fault.cpp)
  target_link_libraries(qwen38_flash_next_snapshot_test PRIVATE
    qwen38_flash_next_test_hip ${CMAKE_DL_LIBS})
  add_executable(qwen38_flash_next_continuation_adapter_test EXCLUDE_FROM_ALL
    continuation_adapter_test.cpp reset_submission_fault.cpp
    device_headroom_fault.cpp)
  target_link_libraries(qwen38_flash_next_continuation_adapter_test PRIVATE
    qwen38_flash_next_test_hip ${CMAKE_DL_LIBS})
  # In-pass checkpoints through the component path against split passes, in
  # AR and MTP modes. Build it with qwen38_flash_next_model_tests. It needs
  # the model: QWEN38_FLASH_NEXT_MODEL names the first GGUF shard and
  # QWEN38_FLASH_NEXT_MTP_MODEL the MTP head. Without them it is skipped.
  add_test(NAME qwen38_flash_next.in_pass_checkpoints
    COMMAND qwen38_flash_next_continuation_adapter_test --in-pass-only)
  set_tests_properties(qwen38_flash_next.in_pass_checkpoints PROPERTIES
    LABELS "qwen38_flash_next;hip;gpu;gfx1151;external-model"
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE
    TIMEOUT 1800)
  # Layered prefill (SetPrefillYield) against lone runs, in AR and MTP modes.
  # Built with qwen38_flash_next_model_tests; needs the same model variables
  # and is skipped without them.
  add_executable(qwen38_flash_next_layered_prefill_test EXCLUDE_FROM_ALL
    layered_prefill_test.cpp)
  target_link_libraries(qwen38_flash_next_layered_prefill_test PRIVATE
    gufo_test_support qwen38_flash_next_test_hip)
  add_test(NAME qwen38_flash_next.layered_prefill
    COMMAND qwen38_flash_next_layered_prefill_test)
  set_tests_properties(qwen38_flash_next.layered_prefill PROPERTIES
    LABELS "qwen38_flash_next;hip;gpu;gfx1151;external-model"
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE
    TIMEOUT 1800)
  add_executable(qwen38_flash_next_continuation_probe EXCLUDE_FROM_ALL
    continuation_probe.cpp)
  target_link_libraries(qwen38_flash_next_continuation_probe PRIVATE
    qwen38_flash_next_test_hip)
  add_custom_target(qwen38_flash_next_model_tests
    DEPENDS qwen38_flash_next_session_test qwen38_flash_next_snapshot_test
      qwen38_flash_next_continuation_adapter_test
      qwen38_flash_next_layered_prefill_test)

  add_executable(qwen38_flash_next_dense_gemm_bench EXCLUDE_FROM_ALL
    dense_gemm_bench.cpp)
  target_link_libraries(qwen38_flash_next_dense_gemm_bench PRIVATE
    qwen38_flash_next_test_hip)

  add_executable(qwen38_flash_next_hc_mix_bench EXCLUDE_FROM_ALL
    hc_mix_bench.cpp)
  target_link_libraries(qwen38_flash_next_hc_mix_bench PRIVATE
    qwen38_flash_next_test_hip)

  add_executable(qwen38_flash_next_attn_bench EXCLUDE_FROM_ALL attn_bench.cpp)
  target_link_libraries(qwen38_flash_next_attn_bench PRIVATE
    qwen38_flash_next_test_hip)

  add_executable(qwen38_flash_next_decode_attn_bench EXCLUDE_FROM_ALL
    decode_attn_bench.cpp)
  target_link_libraries(qwen38_flash_next_decode_attn_bench PRIVATE
    qwen38_flash_next_test_hip)

  add_executable(qwen38_flash_next_routed_gemm_bench EXCLUDE_FROM_ALL
    routed_gemm_bench.cpp)
  target_link_libraries(qwen38_flash_next_routed_gemm_bench PRIVATE
    qwen38_flash_next_test_hip)

  add_executable(qwen38_flash_next_gpu_probe EXCLUDE_FROM_ALL gpu_probe.cpp
    mtp_audit.cpp)
  target_link_libraries(qwen38_flash_next_gpu_probe PRIVATE
    gufo_qwen38_flash_next_reference qwen38_flash_next_test_hip)
endif()

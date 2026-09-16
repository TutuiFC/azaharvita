# citra_common para PS Vita.
# Diferencias con upstream (CMakeLists.upstream.txt):
#   - scm_rev.cpp se genera con valores fijos, sin consultar git
#   - fuera dynamic_library/* : la Vita no tiene dlopen ni ffmpeg
#   - fuera x64/ y aarch64/ : ARMv7

set(GIT_REV    "vita")
set(GIT_BRANCH "vita")
set(GIT_DESC   "azahar-vita")
set(REPO_NAME  "azahar-vita")
set(BUILD_DATE "${CMAKE_CURRENT_LIST_FILE}")
string(TIMESTAMP BUILD_DATE "%Y-%m-%d %H:%M:%S")
set(BUILD_VERSION  "1.0-vita")
set(BUILD_FULLNAME "Azahar Vita 1.0 ")
set(SHADER_CACHE_VERSION "0")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/scm_rev.cpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/scm_rev.cpp" @ONLY)

add_library(citra_common STATIC
    alignment.h
    announce_multiplayer_room.h
    arch.h
    archives.h
    assert.h
    atomic_ops.h
    detached_tasks.cpp
    detached_tasks.h
    bit_field.h
    bit_set.h
    bounded_threadsafe_queue.h
    cityhash.cpp
    cityhash.h
    color.h
    common_funcs.h
    common_paths.h
    common_types.h
    construct.h
    error.cpp
    error.h
    expected.h
    file_derived.cpp
    file_derived.h
    file_util.cpp
    file_util.h
    hash.h
    hacks/hack_list.h
    hacks/hack_list.cpp
    hacks/hack_manager.h
    hacks/hack_manager.cpp
    literals.h
    logging/backend.cpp
    logging/backend.h
    logging/filter.cpp
    logging/filter.h
    logging/formatter.h
    logging/log.h
    logging/log_entry.h
    logging/text_formatter.cpp
    logging/text_formatter.h
    logging/types.h
    math_util.cpp
    math_util.h
    memory_detect.cpp
    memory_detect.h
    memory_ref.h
    memory_ref.cpp
    microprofile.cpp
    microprofile.h
    microprofileui.h
    optional_helper.h
    param_package.cpp
    param_package.h
    play_time_manager.cpp
    play_time_manager.h
    polyfill_thread.h
    quaternion.h
    ring_buffer.h
    "${CMAKE_CURRENT_BINARY_DIR}/scm_rev.cpp"
    scm_rev.h
    scope_exit.h
    setting_keys.h
    settings.cpp
    settings.h
    slot_vector.h
    serialization/atomic.h
    serialization/boost_discrete_interval.hpp
    serialization/boost_flat_set.h
    serialization/boost_small_vector.hpp
    serialization/boost_std_variant.hpp
    serialization/boost_vector.hpp
    static_lru_cache.h
    string_literal.h
    string_util.cpp
    string_util.h
    swap.h
    texture.cpp
    texture.h
    thread.cpp
    thread.h
    thread_queue_list.h
    thread_worker.h
    threadsafe_queue.h
    timer.cpp
    timer.h
    unique_function.h
    vector_math.h
    vita_diag.cpp
    vita_diag.h
    web_util.cpp
    web_util.h
    zstd_compression.cpp
    zstd_compression.h
)

target_include_directories(citra_common PUBLIC ${CMAKE_BINARY_DIR}/src)

target_link_libraries(citra_common PUBLIC fmt library-headers microprofile Boost::boost Boost::serialization)
target_link_libraries(citra_common PRIVATE cryptopp zstd)
target_link_libraries(citra_common PUBLIC xxHash::xxhash)

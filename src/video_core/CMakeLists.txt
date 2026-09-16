add_subdirectory(host_shaders)

add_library(video_core STATIC EXCLUDE_FROM_ALL
    custom_textures/custom_format.cpp
    custom_textures/custom_format.h
    custom_textures/custom_tex_manager.cpp
    custom_textures/custom_tex_manager.h
    custom_textures/material.cpp
    custom_textures/material.h
    debug_utils/debug_utils.cpp
    debug_utils/debug_utils.h
    gpu.cpp
    gpu.h
    gpu_debugger.h
    gpu_impl.h
    pica_types.h
    precompiled_headers.h
    rasterizer_accelerated.cpp
    rasterizer_accelerated.h
    rasterizer_interface.h
    renderer_base.cpp
    renderer_base.h
    pica/dirty_regs.h
    pica/geometry_pipeline.cpp
    pica/geometry_pipeline.h
    pica/pica_core.cpp
    pica/pica_core.h
    pica/output_vertex.cpp
    pica/output_vertex.h
    pica/primitive_assembly.cpp
    pica/primitive_assembly.h
    pica/regs_external.h
    pica/regs_framebuffer.h
    pica/regs_internal.cpp
    pica/regs_internal.h
    pica/regs_lcd.h
    pica/regs_lighting.h
    pica/regs_pipeline.h
    pica/regs_rasterizer.h
    pica/regs_shader.h
    pica/regs_texturing.h
    pica/shader_setup.cpp
    pica/shader_setup.h
    pica/shader_unit.cpp
    pica/shader_unit.h
    pica/packed_attribute.h
    pica/vertex_loader.cpp
    pica/vertex_loader.h
    rasterizer_cache/framebuffer_base.h
    rasterizer_cache/pixel_format.cpp
    rasterizer_cache/pixel_format.h
    rasterizer_cache/rasterizer_cache.cpp
    rasterizer_cache/rasterizer_cache.h
    rasterizer_cache/rasterizer_cache_base.h
    rasterizer_cache/sampler_params.h
    rasterizer_cache/slot_id.h
    rasterizer_cache/surface_base.cpp
    rasterizer_cache/surface_base.h
    rasterizer_cache/surface_params.cpp
    rasterizer_cache/surface_params.h
    rasterizer_cache/texture_codec.h
    rasterizer_cache/texture_cube.h
    rasterizer_cache/utils.cpp
    rasterizer_cache/utils.h
    # Needed as a fallback regardless of enabled renderers.
    renderer_software/sw_blitter.cpp
    renderer_software/sw_blitter.h
    right_eye_disabler.cpp
    right_eye_disabler.h
    shader/debug_data.h
    shader/generator/glsl_fs_shader_gen.cpp
    shader/generator/glsl_fs_shader_gen.h
    shader/generator/glsl_shader_decompiler.cpp
    shader/generator/glsl_shader_decompiler.h
    shader/generator/glsl_shader_gen.cpp
    shader/generator/glsl_shader_gen.h
    shader/generator/pica_fs_config.cpp
    shader/generator/pica_fs_config.h
    shader/generator/profile.h
    shader/generator/shader_gen.cpp
    shader/generator/shader_gen.h
    shader/generator/shader_uniforms.cpp
    shader/generator/shader_uniforms.h
    shader/shader.cpp
    shader/shader.h
    shader/shader_interpreter.cpp
    shader/shader_interpreter.h
    shader/shader_jit.cpp
    shader/shader_jit.h
    shader/shader_jit_a64_compiler.cpp
    shader/shader_jit_a64_compiler.h
    shader/shader_jit_x64_compiler.cpp
    shader/shader_jit_x64_compiler.h
    texture/etc1.cpp
    texture/etc1.h
    texture/texture_decode.cpp
    texture/texture_decode.h
    utils.h
    video_core.cpp
    video_core.h
)

if (ENABLE_SOFTWARE_RENDERER)
    target_sources(video_core PRIVATE
        renderer_software/renderer_software.cpp
        renderer_software/renderer_software.h
        renderer_software/sw_clipper.cpp
        renderer_software/sw_clipper.h
        renderer_software/sw_framebuffer.cpp
        renderer_software/sw_framebuffer.h
        renderer_software/sw_lighting.cpp
        renderer_software/sw_lighting.h
        renderer_software/sw_proctex.cpp
        renderer_software/sw_proctex.h
        renderer_software/sw_rasterizer.cpp
        renderer_software/sw_rasterizer.h
        renderer_software/sw_texturing.cpp
        renderer_software/sw_texturing.h
    )

    # Backend nativo de PS Vita contra sceGxm. En esta fase contiene al de
    # software y lo delega todo (rasterizado incluido); lo nuevo es la
    # presentacion, que dibuja los fotogramas con el chip. Ver
    # renderer_gxm/renderer_gxm.h y renderer_gxm/gxm_presenter.h.
    if (VITA)
        target_sources(video_core PRIVATE
            renderer_gxm/gxm_memory.cpp
            renderer_gxm/gxm_memory.h
            renderer_gxm/gxm_pica_format.h
            renderer_gxm/gxm_presenter.cpp
            renderer_gxm/gxm_presenter.h
            renderer_gxm/rasterizer_gxm.cpp
            renderer_gxm/rasterizer_gxm.h
            renderer_gxm/gxm_texture_cache.cpp
            renderer_gxm/gxm_texture_cache.h
            renderer_gxm/renderer_gxm.cpp
            renderer_gxm/renderer_gxm.h
            # Generador de shaders Cg del backend GXM: el tercer hermano de
            # glsl_fs_shader_gen y spv_fs_shader_gen, mismo PicaFSConfig.
            shader/generator/cg_fs_shader_gen.cpp
            shader/generator/cg_fs_shader_gen.h
        )
    endif()
endif()
if (ENABLE_OPENGL)
    target_sources(video_core PRIVATE
        renderer_opengl/frame_dumper_opengl.cpp
        renderer_opengl/frame_dumper_opengl.h
        renderer_opengl/gl_blit_helper.cpp
        renderer_opengl/gl_blit_helper.h
        renderer_opengl/gl_driver.cpp
        renderer_opengl/gl_driver.h
        renderer_opengl/gl_rasterizer.cpp
        renderer_opengl/gl_rasterizer.h
        renderer_opengl/gl_rasterizer_cache.cpp
        renderer_opengl/gl_resource_manager.cpp
        renderer_opengl/gl_resource_manager.h
        renderer_opengl/gl_shader_disk_cache.cpp
        renderer_opengl/gl_shader_disk_cache.h
        renderer_opengl/gl_shader_manager.cpp
        renderer_opengl/gl_shader_manager.h
        renderer_opengl/gl_shader_util.cpp
        renderer_opengl/gl_shader_util.h
        renderer_opengl/gl_state.cpp
        renderer_opengl/gl_state.h
        renderer_opengl/gl_stream_buffer.cpp
        renderer_opengl/gl_stream_buffer.h
        renderer_opengl/gl_texture_mailbox.cpp
        renderer_opengl/gl_texture_mailbox.h
        renderer_opengl/gl_texture_runtime.cpp
        renderer_opengl/gl_texture_runtime.h
        renderer_opengl/gl_vars.cpp
        renderer_opengl/gl_vars.h
        renderer_opengl/pica_to_gl.h
        renderer_opengl/post_processing_opengl.cpp
        renderer_opengl/post_processing_opengl.h
        renderer_opengl/renderer_opengl.cpp
        renderer_opengl/renderer_opengl.h
    )
    target_link_libraries(video_core PRIVATE glad)
endif()
if (ENABLE_VULKAN)
    target_sources(video_core PRIVATE
        renderer_vulkan/pica_to_vk.h
        renderer_vulkan/renderer_vulkan.cpp
        renderer_vulkan/renderer_vulkan.h
        renderer_vulkan/vk_blit_helper.cpp
        renderer_vulkan/vk_blit_helper.h
        renderer_vulkan/vk_common.cpp
        renderer_vulkan/vk_common.h
        renderer_vulkan/vk_descriptor_update_queue.cpp
        renderer_vulkan/vk_descriptor_update_queue.h
        renderer_vulkan/vk_graphics_pipeline.cpp
        renderer_vulkan/vk_graphics_pipeline.h
        renderer_vulkan/vk_master_semaphore.cpp
        renderer_vulkan/vk_master_semaphore.h
        renderer_vulkan/vk_memory_util.cpp
        renderer_vulkan/vk_memory_util.h
        renderer_vulkan/vk_rasterizer.cpp
        renderer_vulkan/vk_rasterizer.h
        renderer_vulkan/vk_rasterizer_cache.cpp
        renderer_vulkan/vk_scheduler.cpp
        renderer_vulkan/vk_scheduler.h
        renderer_vulkan/vk_resource_pool.cpp
        renderer_vulkan/vk_resource_pool.h
        renderer_vulkan/vk_instance.cpp
        renderer_vulkan/vk_instance.h
        renderer_vulkan/vk_pipeline_cache.cpp
        renderer_vulkan/vk_pipeline_cache.h
        $<$<NOT:$<BOOL:${ENABLE_LIBRETRO}>>:renderer_vulkan/vk_platform.cpp>
        renderer_vulkan/vk_platform.h
        $<$<NOT:$<BOOL:${ENABLE_LIBRETRO}>>:renderer_vulkan/vk_present_window.cpp>
        $<$<NOT:$<BOOL:${ENABLE_LIBRETRO}>>:renderer_vulkan/vk_present_window.h>
        renderer_vulkan/vk_render_manager.cpp
        renderer_vulkan/vk_render_manager.h
        renderer_vulkan/vk_shader_disk_cache.cpp
        renderer_vulkan/vk_shader_disk_cache.h
        renderer_vulkan/vk_shader_util.cpp
        renderer_vulkan/vk_shader_util.h
        renderer_vulkan/vk_stream_buffer.cpp
        renderer_vulkan/vk_stream_buffer.h
        $<$<NOT:$<BOOL:${ENABLE_LIBRETRO}>>:renderer_vulkan/vk_swapchain.cpp>
        $<$<NOT:$<BOOL:${ENABLE_LIBRETRO}>>:renderer_vulkan/vk_swapchain.h>
        renderer_vulkan/vk_texture_runtime.cpp
        renderer_vulkan/vk_texture_runtime.h
        shader/generator/spv_fs_shader_gen.cpp
        shader/generator/spv_fs_shader_gen.h
    )
    target_link_libraries(video_core PRIVATE vulkan-headers vma sirit SPIRV glslang)
endif()

if(ENABLE_SDL2)
    target_link_libraries(video_core PRIVATE SDL2::SDL2)
endif()

add_dependencies(video_core host_shaders)
target_include_directories(video_core PRIVATE ${HOST_SHADERS_INCLUDE})

create_target_directory_groups(video_core)

target_link_libraries(video_core PUBLIC citra_common citra_core)
target_link_libraries(video_core PRIVATE Boost::serialization dds-ktx json-headers nihstro-headers tsl::robin_map)

if ("x86_64" IN_LIST ARCHITECTURE)
    target_link_libraries(video_core PUBLIC xbyak)
endif()

if ("arm64" IN_LIST ARCHITECTURE)
    target_link_libraries(video_core PUBLIC oaknut)
endif()

if (CITRA_USE_PRECOMPILED_HEADERS)
    target_precompile_headers(video_core PRIVATE precompiled_headers.h)
endif()

# Igual que con el interprete de CPU (ver el comentario en src/core/CMakeLists.txt):
# el rasterizador software dibuja pixel a pixel y el interprete de shaders de la
# PICA200 corre una vez por vertice, y son el segundo y tercer consumidor de
# tiempo mas grande despues de la CPU. -O3 para que el auto-vectorizador de GCC
# pueda usar el NEON que el toolchain ya trae activado por defecto.
set_source_files_properties(
    renderer_software/sw_rasterizer.cpp
    renderer_software/sw_blitter.cpp
    renderer_software/renderer_software.cpp
    renderer_software/sw_framebuffer.cpp
    renderer_software/sw_texturing.cpp
    renderer_software/sw_clipper.cpp
    renderer_software/sw_lighting.cpp
    renderer_software/sw_proctex.cpp
    shader/shader_interpreter.cpp
    shader/shader.cpp
    pica/pica_core.cpp

    # Estos dos faltaban, y son de los mas calientes que hay.
    #
    # LookupTexture/LookupTexelInTile corren una vez POR PIXEL Y POR UNIDAD de
    # textura (hasta tres veces por pixel), y SampleETC1Subtile ademas
    # descomprime el bloque ETC1 en cada texel -- que es el formato en el que
    # vienen casi todas las texturas de 3DS. Se estaban compilando a -Os junto
    # con el codigo de arranque: sin desplegar el switch de formatos y sin meter
    # en linea el entrelazado de Morton ni los decodificadores de color.
    texture/texture_decode.cpp
    texture/etc1.cpp
    PROPERTIES COMPILE_OPTIONS "-O3;-flto"
)

# Por que -flto ademas de -O3.
#
# -O3 optimiza cada fichero por separado, y el problema del rasterizador no esta
# dentro de ninguno: esta en las fronteras entre ellos. Contando las llamadas que
# hace el bucle de pixeles a otras unidades de compilacion salen entre 60 y 80
# POR PIXEL:
#
#   GetColorModifier / GetAlphaModifier   3 fuentes x 6 etapas TEV x 2 = 36
#   ColorCombine / AlphaCombine                                       = 12
#   GetWrappedTexCoord                                                =  6
#   LookupTexture                                                     =  3
#   Framebuffer Get/Set de color, profundidad y galga                 = ~5
#   EvaluateBlendEquation, ComputeFragmentsColors                     =  2
#
# Y el coste no es solo el salto. La configuracion TEV no cambia en todo el
# triangulo, pero como el switch vive al otro lado de una llamada, el compilador
# no puede resolverlo: lo vuelve a recorrer entero 36 veces por pixel para
# elegir siempre la misma rama. Con LTO ve las dos partes a la vez, mete las
# funciones en linea y propaga esas constantes dentro del bucle.
#
# Hace falta que TODAS las partes implicadas se compilen con -flto, de ahi que
# la lista de arriba sea la misma. gcc-ar/gcc-ranlib son necesarios para que la
# biblioteca estatica lleve bien el indice de simbolos (ver el CMakeLists raiz).

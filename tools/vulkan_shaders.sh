#!/bin/sh
# Recompiles the Vulkan backend's compute shaders (src/vulkan/*.comp) into the SPIR-V that is
# checked in next to them and embedded in the binary (src/vulkan.rs), so that building oxinfer
# needs no shader compiler. Needs glslc (shaderc); run from the repository root after editing
# a shader, and commit the .spv files with it.
set -e
for f in bf16 q8 q4 q4a; do
    def=$(echo "$f" | tr a-z A-Z)
    glslc -O --target-env=vulkan1.0 -fshader-stage=compute "-DFMT_$def" src/vulkan/gemm.comp -o "src/vulkan/gemm_$f.spv"
done

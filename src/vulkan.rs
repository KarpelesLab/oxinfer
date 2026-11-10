//! Vulkan compute backend (any GPU with a Vulkan driver: Linux, Windows, Android, MoltenVK): the
//! Vulkan loader is opened at run time with `dlsym`, so the binary has no Vulkan dependency and
//! runs unchanged on machines without it. The shaders are GLSL compute sources in `src/vulkan/`
//! with their compiled SPIR-V checked in next to them and embedded here, so nothing is needed at
//! build time either; `tools/vulkan_shaders.sh` regenerates the SPIR-V (it needs `glslc`).
//!
//! Only the prefill-sized matmuls are offloaded (see `wants`): bf16, Q8, Q4 and affine Q4 weights.
//! Convolutions, attention, MoE experts and decode steps stay on the CPU with this backend.
//!
//! Only Vulkan 1.0 core features are required (no extension, no 8- or 16-bit storage access:
//! the shaders read the weights as 32-bit words), plus compute workgroups of 256 invocations
//! with 16 KiB of shared memory, which the backend checks when it starts.
//!
//! Memory: nothing assumes unified memory. On discrete (and virtual) devices the weights are
//! copied once into device-local buffers through a host-visible staging buffer, and each batch
//! copies the activations up and the results down inside its submission. On integrated devices
//! the buffers are host-visible and the shaders read and write them directly
//! (`OXINFER_VULKAN_STAGED=0|1` forces either mode). A weight matrix larger than the device's
//! `maxStorageBufferRange` or `maxMemoryAllocationSize` is split by rows into several buffers
//! (one dispatch each), and a batch whose activations exceed them is run in several token
//! ranges. A matrix that does not fit in device memory is declined (that layer stays on the
//! CPU); with `VK_EXT_memory_budget` the heap's budget is checked first, since some drivers
//! overcommit instead of failing.
//!
//! Every output is one accumulator taking one fused multiply-add per input in order, the CPU
//! prefill GEMM's arithmetic; whether the results are bit-identical to the CPU's depends on the
//! driver compiling `fma` to a fused operation (see the README for what was measured).
//! `cargo test --lib vulkan::` compares each format with references computed from its
//! definition, in both memory modes; the tests skip on machines without a Vulkan device.

use crate::gpu::{Backend, DeviceWeight, Job, WeightRef};
use anyhow::{Result, anyhow, bail};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const TILE: usize = 64;
/// Smaller token counts stay on the CPU.
const MIN_T: usize = 16;
/// Smallest matmul worth a submission: the CUDA backend's threshold where every batch is copied
/// to the device and back, Metal's where the buffers are shared. Not tuned for Vulkan.
const MIN_FLOPS_STAGED: f64 = 1e9;
const MIN_FLOPS_DIRECT: f64 = 3e8;
/// Device memory left free when the driver reports its budget.
const VRAM_RESERVE: u64 = 512 << 20;
/// Size of the staging buffer weights are uploaded through.
const UPLOAD_STAGE: u64 = 32 << 20;
/// Descriptor sets (one per dispatch of a submission) per pool.
const SETS_PER_POOL: u32 = 64;

static SPIRV: [&[u8]; 4] = [
    include_bytes!("vulkan/gemm_bf16.spv"),
    include_bytes!("vulkan/gemm_q8.spv"),
    include_bytes!("vulkan/gemm_q4.spv"),
    include_bytes!("vulkan/gemm_q4a.spv"),
];

// Dispatchable handles are pointers; the others are 64-bit on every platform.
type Ptr = *mut c_void;
type H = u64;

// Structure types, flags and enums used below (vulkan_core.h).
const ST_APPLICATION_INFO: u32 = 0;
const ST_INSTANCE_CREATE_INFO: u32 = 1;
const ST_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
const ST_DEVICE_CREATE_INFO: u32 = 3;
const ST_SUBMIT_INFO: u32 = 4;
const ST_MEMORY_ALLOCATE_INFO: u32 = 5;
const ST_BUFFER_CREATE_INFO: u32 = 12;
const ST_SHADER_MODULE_CREATE_INFO: u32 = 16;
const ST_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
const ST_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
const ST_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
const ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
const ST_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
const ST_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
const ST_WRITE_DESCRIPTOR_SET: u32 = 35;
const ST_COMMAND_POOL_CREATE_INFO: u32 = 39;
const ST_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
const ST_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
const ST_MEMORY_BARRIER: u32 = 46;
const ST_PHYSICAL_DEVICE_PROPERTIES_2: u32 = 1000059001;
const ST_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2: u32 = 1000059006;
const ST_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES: u32 = 1000168000;
const ST_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES: u32 = 1000237000;

const API_1_0: u32 = 1 << 22;
const API_1_1: u32 = (1 << 22) | (1 << 12);
const QUEUE_COMPUTE: u32 = 2;
const MEM_DEVICE_LOCAL: u32 = 1;
const MEM_HOST_VISIBLE: u32 = 2;
const MEM_HOST_COHERENT: u32 = 4;
const MEM_HOST_CACHED: u32 = 8;
const USAGE_TRANSFER_SRC: u32 = 1;
const USAGE_TRANSFER_DST: u32 = 2;
const USAGE_STORAGE: u32 = 32;
const DESCRIPTOR_STORAGE_BUFFER: u32 = 7;
const STAGE_COMPUTE: u32 = 32;
const BIND_POINT_COMPUTE: u32 = 1;
const PIPELINE_STAGE_COMPUTE: u32 = 2048;
const PIPELINE_STAGE_TRANSFER: u32 = 4096;
const ACCESS_SHADER_READ: u32 = 32;
const ACCESS_SHADER_WRITE: u32 = 64;
const ACCESS_TRANSFER_READ: u32 = 2048;
const ACCESS_TRANSFER_WRITE: u32 = 4096;
const COMMAND_POOL_RESET_COMMAND_BUFFER: u32 = 2;
const COMMAND_BUFFER_ONE_TIME_SUBMIT: u32 = 1;
const DEVICE_TYPE_INTEGRATED: u32 = 1;
const DEVICE_TYPE_CPU: u32 = 4;

#[repr(C)]
pub struct ApplicationInfo {
    s_type: u32,
    p_next: *const c_void,
    app_name: *const c_char,
    app_version: u32,
    engine_name: *const c_char,
    engine_version: u32,
    api_version: u32,
}
#[repr(C)]
pub struct InstanceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    app_info: *const ApplicationInfo,
    layer_count: u32,
    layers: *const *const c_char,
    ext_count: u32,
    exts: *const *const c_char,
}
#[repr(C)]
pub struct DeviceQueueCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_family_index: u32,
    queue_count: u32,
    priorities: *const f32,
}
#[repr(C)]
pub struct DeviceCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_info_count: u32,
    queue_infos: *const DeviceQueueCreateInfo,
    layer_count: u32,
    layers: *const *const c_char,
    ext_count: u32,
    exts: *const *const c_char,
    features: *const c_void,
}
#[repr(C)]
pub struct BufferCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    size: u64,
    usage: u32,
    sharing_mode: u32,
    queue_family_count: u32,
    queue_families: *const u32,
}
#[repr(C)]
#[derive(Default)]
pub struct MemoryRequirements {
    size: u64,
    alignment: u64,
    memory_type_bits: u32,
}
#[repr(C)]
pub struct MemoryAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    size: u64,
    memory_type_index: u32,
}
#[repr(C)]
pub struct ShaderModuleCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    code_size: usize,
    code: *const u32,
}
#[repr(C)]
pub struct DescriptorSetLayoutBinding {
    binding: u32,
    descriptor_type: u32,
    descriptor_count: u32,
    stage_flags: u32,
    immutable_samplers: *const c_void,
}
#[repr(C)]
pub struct DescriptorSetLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    binding_count: u32,
    bindings: *const DescriptorSetLayoutBinding,
}
#[repr(C)]
pub struct PushConstantRange {
    stage_flags: u32,
    offset: u32,
    size: u32,
}
#[repr(C)]
pub struct PipelineLayoutCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    set_layout_count: u32,
    set_layouts: *const H,
    push_constant_range_count: u32,
    push_constant_ranges: *const PushConstantRange,
}
#[repr(C)]
pub struct PipelineShaderStageCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage: u32,
    module: H,
    name: *const c_char,
    specialization_info: *const c_void,
}
#[repr(C)]
pub struct ComputePipelineCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    stage: PipelineShaderStageCreateInfo,
    layout: H,
    base_pipeline: H,
    base_pipeline_index: i32,
}
#[repr(C)]
pub struct DescriptorPoolSize {
    ty: u32,
    count: u32,
}
#[repr(C)]
pub struct DescriptorPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    max_sets: u32,
    pool_size_count: u32,
    pool_sizes: *const DescriptorPoolSize,
}
#[repr(C)]
pub struct DescriptorSetAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    pool: H,
    count: u32,
    set_layouts: *const H,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DescriptorBufferInfo {
    buffer: H,
    offset: u64,
    range: u64,
}
#[repr(C)]
pub struct WriteDescriptorSet {
    s_type: u32,
    p_next: *const c_void,
    dst_set: H,
    dst_binding: u32,
    dst_array_element: u32,
    descriptor_count: u32,
    descriptor_type: u32,
    image_info: *const c_void,
    buffer_info: *const DescriptorBufferInfo,
    texel_buffer_view: *const c_void,
}
#[repr(C)]
pub struct CommandPoolCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    queue_family_index: u32,
}
#[repr(C)]
pub struct CommandBufferAllocateInfo {
    s_type: u32,
    p_next: *const c_void,
    pool: H,
    level: u32,
    count: u32,
}
#[repr(C)]
pub struct CommandBufferBeginInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    inheritance: *const c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BufferCopy {
    src_offset: u64,
    dst_offset: u64,
    size: u64,
}
#[repr(C)]
pub struct SubmitInfo {
    s_type: u32,
    p_next: *const c_void,
    wait_semaphore_count: u32,
    wait_semaphores: *const H,
    wait_dst_stage_mask: *const u32,
    command_buffer_count: u32,
    command_buffers: *const Ptr,
    signal_semaphore_count: u32,
    signal_semaphores: *const H,
}
#[repr(C)]
pub struct MemoryBarrier {
    s_type: u32,
    p_next: *const c_void,
    src_access: u32,
    dst_access: u32,
}
/// The gemm shaders' push constants (`src/vulkan/gemm.comp`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Push {
    n_out: u32,
    n_in: u32,
    row0: u32,
    s_base: u32,
    o_base: u32,
    gpr: u32,
    gk: u32,
}

// The loader exports every core command, so they are resolved by name like any C library's.
// (`extern "C"`: VKAPI_CALL is the C convention on every 64-bit target and on 32-bit ARM.)
dlsym::library! {
    pub struct Vk {
        unsafe extern "C" {
            #[optional]
            pub fn vkEnumerateInstanceVersion(v: *mut u32) -> c_int;
            pub fn vkCreateInstance(info: *const InstanceCreateInfo, alloc: *const c_void, out: *mut Ptr) -> c_int;
            pub fn vkDestroyInstance(inst: Ptr, alloc: *const c_void);
            pub fn vkEnumeratePhysicalDevices(inst: Ptr, count: *mut u32, devs: *mut Ptr) -> c_int;
            pub fn vkGetPhysicalDeviceProperties(pd: Ptr, props: *mut u8);
            #[optional]
            pub fn vkGetPhysicalDeviceProperties2(pd: Ptr, props: *mut u8);
            pub fn vkGetPhysicalDeviceMemoryProperties(pd: Ptr, props: *mut u8);
            #[optional]
            pub fn vkGetPhysicalDeviceMemoryProperties2(pd: Ptr, props: *mut u8);
            pub fn vkGetPhysicalDeviceQueueFamilyProperties(pd: Ptr, count: *mut u32, props: *mut u8);
            pub fn vkEnumerateDeviceExtensionProperties(pd: Ptr, layer: *const c_char, count: *mut u32, props: *mut u8) -> c_int;
            pub fn vkCreateDevice(pd: Ptr, info: *const DeviceCreateInfo, alloc: *const c_void, out: *mut Ptr) -> c_int;
            pub fn vkDestroyDevice(dev: Ptr, alloc: *const c_void);
            pub fn vkGetDeviceQueue(dev: Ptr, family: u32, index: u32, out: *mut Ptr);
            pub fn vkDeviceWaitIdle(dev: Ptr) -> c_int;
            pub fn vkCreateBuffer(dev: Ptr, info: *const BufferCreateInfo, alloc: *const c_void, out: *mut H) -> c_int;
            pub fn vkDestroyBuffer(dev: Ptr, buf: H, alloc: *const c_void);
            pub fn vkGetBufferMemoryRequirements(dev: Ptr, buf: H, out: *mut MemoryRequirements);
            pub fn vkAllocateMemory(dev: Ptr, info: *const MemoryAllocateInfo, alloc: *const c_void, out: *mut H) -> c_int;
            pub fn vkFreeMemory(dev: Ptr, mem: H, alloc: *const c_void);
            pub fn vkBindBufferMemory(dev: Ptr, buf: H, mem: H, offset: u64) -> c_int;
            pub fn vkMapMemory(dev: Ptr, mem: H, offset: u64, size: u64, flags: u32, out: *mut *mut c_void) -> c_int;
            pub fn vkUnmapMemory(dev: Ptr, mem: H);
            pub fn vkCreateShaderModule(dev: Ptr, info: *const ShaderModuleCreateInfo, alloc: *const c_void, out: *mut H) -> c_int;
            pub fn vkDestroyShaderModule(dev: Ptr, module: H, alloc: *const c_void);
            pub fn vkCreateDescriptorSetLayout(dev: Ptr, info: *const DescriptorSetLayoutCreateInfo, alloc: *const c_void, out: *mut H) -> c_int;
            pub fn vkDestroyDescriptorSetLayout(dev: Ptr, layout: H, alloc: *const c_void);
            pub fn vkCreatePipelineLayout(dev: Ptr, info: *const PipelineLayoutCreateInfo, alloc: *const c_void, out: *mut H) -> c_int;
            pub fn vkDestroyPipelineLayout(dev: Ptr, layout: H, alloc: *const c_void);
            pub fn vkCreateComputePipelines(
                dev: Ptr, cache: H, count: u32, infos: *const ComputePipelineCreateInfo, alloc: *const c_void, out: *mut H,
            ) -> c_int;
            pub fn vkDestroyPipeline(dev: Ptr, pipeline: H, alloc: *const c_void);
            pub fn vkCreateDescriptorPool(dev: Ptr, info: *const DescriptorPoolCreateInfo, alloc: *const c_void, out: *mut H) -> c_int;
            pub fn vkDestroyDescriptorPool(dev: Ptr, pool: H, alloc: *const c_void);
            pub fn vkAllocateDescriptorSets(dev: Ptr, info: *const DescriptorSetAllocateInfo, out: *mut H) -> c_int;
            pub fn vkUpdateDescriptorSets(dev: Ptr, n_writes: u32, writes: *const WriteDescriptorSet, n_copies: u32, copies: *const c_void);
            pub fn vkCreateCommandPool(dev: Ptr, info: *const CommandPoolCreateInfo, alloc: *const c_void, out: *mut H) -> c_int;
            pub fn vkDestroyCommandPool(dev: Ptr, pool: H, alloc: *const c_void);
            pub fn vkAllocateCommandBuffers(dev: Ptr, info: *const CommandBufferAllocateInfo, out: *mut Ptr) -> c_int;
            pub fn vkBeginCommandBuffer(cmd: Ptr, info: *const CommandBufferBeginInfo) -> c_int;
            pub fn vkEndCommandBuffer(cmd: Ptr) -> c_int;
            pub fn vkCmdBindPipeline(cmd: Ptr, bind_point: u32, pipeline: H);
            pub fn vkCmdBindDescriptorSets(
                cmd: Ptr, bind_point: u32, layout: H, first: u32, count: u32, sets: *const H, n_dynamic: u32, dynamic: *const u32,
            );
            pub fn vkCmdPushConstants(cmd: Ptr, layout: H, stages: u32, offset: u32, size: u32, values: *const c_void);
            pub fn vkCmdDispatch(cmd: Ptr, x: u32, y: u32, z: u32);
            pub fn vkCmdCopyBuffer(cmd: Ptr, src: H, dst: H, count: u32, regions: *const BufferCopy);
            pub fn vkCmdPipelineBarrier(
                cmd: Ptr, src_stages: u32, dst_stages: u32, dependency_flags: u32,
                n_memory: u32, memory: *const MemoryBarrier,
                n_buffer: u32, buffer: *const c_void, n_image: u32, image: *const c_void,
            );
            pub fn vkQueueSubmit(queue: Ptr, count: u32, submits: *const SubmitInfo, fence: H) -> c_int;
            pub fn vkQueueWaitIdle(queue: Ptr) -> c_int;
        }
    }
}

fn check(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let text = match rc {
        -1 => "out of host memory",
        -2 => "out of device memory",
        -3 => "initialization failed",
        -4 => "device lost",
        -7 => "extension not present",
        -8 => "feature not present",
        -9 => "incompatible driver",
        _ => bail!("{what}: error {rc}"),
    };
    Err(anyhow!("{what}: {text}"))
}

fn rd32(b: &[u8], at: usize) -> u32 {
    u32::from_ne_bytes(b[at..at + 4].try_into().unwrap())
}

/// A buffer with its own memory; `ptr` is the mapping when it is mapped.
struct Buf {
    buffer: H,
    memory: H,
    size: u64,
    ptr: *mut u8,
}

impl Default for Buf {
    fn default() -> Self {
        Buf {
            buffer: 0,
            memory: 0,
            size: 0,
            ptr: std::ptr::null_mut(),
        }
    }
}

/// Where a buffer's memory should live, in order of preference (see [`Core::pick_type`]).
#[derive(Clone, Copy, PartialEq)]
enum Place {
    /// Device memory the host need not see (weights and activations of a discrete device).
    Device,
    /// Host-visible memory the device only copies from or to.
    Host,
    /// Host-visible memory the shaders work in (integrated devices): device-local if possible.
    Shared,
}

/// The instance and device, shared with the weights (which free their buffers on drop).
struct Core {
    vk: Vk,
    inst: Ptr,
    pd: Ptr,
    dev: Ptr,
    queue: Ptr,
    /// (property flags, heap) of each memory type.
    mem_types: Vec<(u32, u32)>,
    /// VK_EXT_memory_budget is enabled.
    budget: bool,
    allocs: AtomicU32,
    max_allocs: u32,
}
unsafe impl Send for Core {}
unsafe impl Sync for Core {}

impl Drop for Core {
    fn drop(&mut self) {
        unsafe {
            if !self.dev.is_null() {
                self.vk.vkDeviceWaitIdle(self.dev);
                self.vk.vkDestroyDevice(self.dev, std::ptr::null());
            }
            self.vk.vkDestroyInstance(self.inst, std::ptr::null());
        }
    }
}

impl Core {
    /// The first memory type among `bits` (a buffer's `memoryTypeBits`) with the properties of
    /// the earliest possible preference of `place`.
    fn pick_type(&self, bits: u32, place: Place) -> Option<u32> {
        const HOST: u32 = MEM_HOST_VISIBLE | MEM_HOST_COHERENT;
        // (required, excluded)
        let prefs: &[(u32, u32)] = match place {
            Place::Device => &[(MEM_DEVICE_LOCAL, MEM_HOST_VISIBLE), (MEM_DEVICE_LOCAL, 0), (0, 0)],
            // Cached memory first: results are read back from it.
            Place::Host => &[(HOST | MEM_HOST_CACHED, 0), (HOST, 0)],
            Place::Shared => &[
                (HOST | MEM_HOST_CACHED | MEM_DEVICE_LOCAL, 0),
                (HOST | MEM_HOST_CACHED, 0),
                (HOST | MEM_DEVICE_LOCAL, 0),
                (HOST, 0),
            ],
        };
        for &(want, avoid) in prefs {
            for (i, &(flags, _)) in self.mem_types.iter().enumerate() {
                if bits & (1 << i) != 0 && flags & want == want && flags & avoid == 0 {
                    return Some(i as u32);
                }
            }
        }
        None
    }

    /// Bytes the heap of memory type `ty` can still take within its budget, when the driver
    /// reports budgets.
    fn heap_room(&self, ty: u32) -> Option<u64> {
        if !self.budget {
            return None;
        }
        // VkPhysicalDeviceMemoryProperties2 (536 bytes) chained to
        // VkPhysicalDeviceMemoryBudgetPropertiesEXT (272: heapBudget[16] @16, heapUsage[16] @144).
        let mut budget = [0u64; 34];
        let mut props = [0u64; 67];
        budget[0] = ST_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES as u64;
        props[0] = ST_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2 as u64;
        props[1] = budget.as_mut_ptr() as u64;
        unsafe { self.vk.vkGetPhysicalDeviceMemoryProperties2(self.pd, props.as_mut_ptr() as *mut u8) }?;
        let heap = self.mem_types[ty as usize].1 as usize;
        let (b, used) = (budget[2 + heap], budget[18 + heap]);
        (b > 0).then(|| b.saturating_sub(used))
    }

    /// A buffer of `size` bytes with its own memory, mapped if `map`. `reserve`: bytes its heap
    /// must keep free after the allocation where the driver reports budgets. None when the
    /// device has no room (or no suitable memory).
    fn create_buf(&self, size: u64, usage: u32, place: Place, map: bool, reserve: Option<u64>) -> Option<Buf> {
        if self.allocs.load(Ordering::Relaxed) >= self.max_allocs {
            return None;
        }
        let info = BufferCreateInfo {
            s_type: ST_BUFFER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            size,
            usage,
            sharing_mode: 0,
            queue_family_count: 0,
            queue_families: std::ptr::null(),
        };
        let mut b = Buf { size, ..Buf::default() };
        if unsafe { self.vk.vkCreateBuffer(self.dev, &info, std::ptr::null(), &mut b.buffer) } != 0 {
            return None;
        }
        let mut req = MemoryRequirements::default();
        unsafe { self.vk.vkGetBufferMemoryRequirements(self.dev, b.buffer, &mut req) };
        let ok = (|| {
            let ty = self.pick_type(req.memory_type_bits, place)?;
            if let (Some(reserve), Some(room)) = (reserve, self.heap_room(ty))
                && room < req.size.saturating_add(reserve)
            {
                return None;
            }
            let alloc = MemoryAllocateInfo {
                s_type: ST_MEMORY_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                size: req.size,
                memory_type_index: ty,
            };
            if unsafe { self.vk.vkAllocateMemory(self.dev, &alloc, std::ptr::null(), &mut b.memory) } != 0 {
                b.memory = 0;
                return None;
            }
            self.allocs.fetch_add(1, Ordering::Relaxed);
            if unsafe { self.vk.vkBindBufferMemory(self.dev, b.buffer, b.memory, 0) } != 0 {
                return None;
            }
            if map {
                let mut p = std::ptr::null_mut();
                if unsafe { self.vk.vkMapMemory(self.dev, b.memory, 0, !0, 0, &mut p) } != 0 || p.is_null() {
                    return None;
                }
                b.ptr = p as *mut u8;
            }
            Some(())
        })();
        if ok.is_none() {
            self.destroy_buf(&mut b);
            return None;
        }
        Some(b)
    }

    /// Frees a buffer and its memory (unmapping it); the buffer must not be in use by a pending
    /// submission.
    fn destroy_buf(&self, b: &mut Buf) {
        unsafe {
            if b.buffer != 0 {
                self.vk.vkDestroyBuffer(self.dev, b.buffer, std::ptr::null());
            }
            if b.memory != 0 {
                self.vk.vkFreeMemory(self.dev, b.memory, std::ptr::null());
                self.allocs.fetch_sub(1, Ordering::Relaxed);
            }
        }
        *b = Buf::default();
    }
}

/// Whole rows `row0 .. row0 + rows` of a weight matrix in one buffer: the values, the scales
/// from byte `s_off` and the affine offsets from byte `o_off`.
struct Chunk {
    buf: Buf,
    row0: usize,
    rows: usize,
    s_off: u64,
    o_off: u64,
}

/// A weight matrix on the device, freed when its layer is dropped.
struct VkWeight {
    core: Arc<Core>,
    /// Index of the format's pipeline.
    fmt: usize,
    n_out: usize,
    n_in: usize,
    gpr: u32,
    gk: u32,
    chunks: Vec<Chunk>,
}
unsafe impl Send for VkWeight {}
unsafe impl Sync for VkWeight {}

impl Drop for VkWeight {
    fn drop(&mut self) {
        // Submissions are synchronous and borrow the weight, so none is pending.
        for c in &mut self.chunks {
            self.core.destroy_buf(&mut c.buf);
        }
    }
}

/// What a submission needs exclusively: the command buffer, the descriptor sets, the staging
/// buffers.
struct State {
    cmd: Ptr,
    pools: Vec<H>,
    sets: Vec<H>,
    /// Activations and results in host-visible memory (mapped), and on the device when staged.
    xh: Buf,
    yh: Buf,
    xd: Buf,
    yd: Buf,
    /// Weights on their way to device-local buffers (mapped).
    up: Buf,
}

#[derive(Default)]
struct Stats {
    calls: AtomicU64,
    copy_in_ns: AtomicU64,
    gpu_ns: AtomicU64,
    copy_out_ns: AtomicU64,
}

pub struct VulkanBackend {
    core: Arc<Core>,
    st: Mutex<State>,
    set_layout: H,
    layout: H,
    modules: [H; 4],
    /// bf16, q8, q4, affine q4.
    pipelines: [H; 4],
    cmd_pool: H,
    /// Weights in device-local memory behind a staging buffer, activations copied up and down
    /// per batch; otherwise everything in host-visible memory the shaders use directly.
    staged: bool,
    /// Largest storage-buffer binding and largest allocation, in bytes.
    range_limit: u64,
    alloc_limit: u64,
    /// Workgroups per dispatch along x (rows) and y (tokens).
    max_groups: [u32; 2],
    stats: Stats,
    name: String,
}
unsafe impl Send for VulkanBackend {}
unsafe impl Sync for VulkanBackend {}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        let c = &self.core;
        let st = self.st.get_mut().unwrap_or_else(|e| e.into_inner());
        unsafe {
            c.vk.vkQueueWaitIdle(c.queue);
            for b in [&mut st.xh, &mut st.yh, &mut st.xd, &mut st.yd, &mut st.up] {
                c.destroy_buf(b);
            }
            for &p in &st.pools {
                c.vk.vkDestroyDescriptorPool(c.dev, p, std::ptr::null());
            }
            for &p in &self.pipelines {
                c.vk.vkDestroyPipeline(c.dev, p, std::ptr::null());
            }
            for &m in &self.modules {
                c.vk.vkDestroyShaderModule(c.dev, m, std::ptr::null());
            }
            c.vk.vkDestroyPipelineLayout(c.dev, self.layout, std::ptr::null());
            c.vk.vkDestroyDescriptorSetLayout(c.dev, self.set_layout, std::ptr::null());
            // Frees the command buffer with it.
            c.vk.vkDestroyCommandPool(c.dev, self.cmd_pool, std::ptr::null());
        }
    }
}

impl VulkanBackend {
    /// The backend on physical device `index` (the loader's enumeration order, as in
    /// [`crate::platform::GpuInfo::index`]).
    pub fn new(index: usize) -> Result<Self> {
        let staged = match std::env::var("OXINFER_VULKAN_STAGED").ok().as_deref() {
            Some("0") => Some(false),
            Some("1") => Some(true),
            _ => None,
        };
        Self::with_mode(index, staged)
    }

    /// [`Self::new`] with the memory mode forced (`staged`), or chosen from the device type.
    fn with_mode(index: usize, staged: Option<bool>) -> Result<Self> {
        let vk = {
            use dlsym::Bindings;
            // SAFETY: the loader's initialisers have no preconditions.
            unsafe { Vk::open_first(crate::platform::gpu::vulkan::CANDIDATES.iter().copied()) }.map_err(|e| anyhow!("libvulkan: {e}"))?
        };
        // A 1.1 instance where the loader has one (for the properties2 queries below).
        let mut inst_ver = API_1_0;
        if unsafe { vk.vkEnumerateInstanceVersion(&mut inst_ver) } != Some(0) {
            inst_ver = API_1_0;
        }
        let v11 = inst_ver >= API_1_1;
        let app = ApplicationInfo {
            s_type: ST_APPLICATION_INFO,
            p_next: std::ptr::null(),
            app_name: c"oxinfer".as_ptr(),
            app_version: 0,
            engine_name: c"oxinfer".as_ptr(),
            engine_version: 0,
            api_version: if v11 { API_1_1 } else { API_1_0 },
        };
        let info = InstanceCreateInfo {
            s_type: ST_INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            app_info: &app,
            layer_count: 0,
            layers: std::ptr::null(),
            ext_count: 0,
            exts: std::ptr::null(),
        };
        let mut inst: Ptr = std::ptr::null_mut();
        check(
            unsafe { vk.vkCreateInstance(&info, std::ptr::null(), &mut inst) },
            "vkCreateInstance",
        )?;
        // From here on `core` owns the instance (and the device once it exists).
        let mut core = Core {
            vk,
            inst,
            pd: std::ptr::null_mut(),
            dev: std::ptr::null_mut(),
            queue: std::ptr::null_mut(),
            mem_types: Vec::new(),
            budget: false,
            allocs: AtomicU32::new(0),
            max_allocs: 0,
        };
        let vk = &core.vk;

        let mut count = 0u32;
        check(
            unsafe { vk.vkEnumeratePhysicalDevices(inst, &mut count, std::ptr::null_mut()) },
            "vkEnumeratePhysicalDevices",
        )?;
        let mut pds = vec![std::ptr::null_mut(); count as usize];
        let rc = unsafe { vk.vkEnumeratePhysicalDevices(inst, &mut count, pds.as_mut_ptr()) };
        if rc < 0 {
            check(rc, "vkEnumeratePhysicalDevices")?;
        }
        let Some(&pd) = pds.get(index).filter(|_| index < count as usize) else {
            bail!("no physical device #{index} ({count} found)");
        };

        // VkPhysicalDeviceProperties (824 bytes): apiVersion @0, deviceType @16, deviceName @20,
        // limits @296.
        let mut props = vec![0u8; 1024];
        unsafe { vk.vkGetPhysicalDeviceProperties(pd, props.as_mut_ptr()) };
        let dev_api = rd32(&props, 0);
        let dev_type = rd32(&props, 16);
        let dev_name = CStr::from_bytes_until_nul(&props[20..276]).map_or(String::new(), |s| s.to_string_lossy().into_owned());
        const LIMITS: usize = 296;
        let max_range = rd32(&props, LIMITS + 28) as u64; // maxStorageBufferRange
        let max_push = rd32(&props, LIMITS + 32); // maxPushConstantsSize
        let max_allocs = rd32(&props, LIMITS + 36); // maxMemoryAllocationCount
        let max_storage = rd32(&props, LIMITS + 76); // maxPerStageDescriptorStorageBuffers
        let shared_mem = rd32(&props, LIMITS + 216); // maxComputeSharedMemorySize
        let max_groups = [rd32(&props, LIMITS + 220), rd32(&props, LIMITS + 224)]; // maxComputeWorkGroupCount
        let max_invocations = rd32(&props, LIMITS + 232); // maxComputeWorkGroupInvocations
        let max_local_x = rd32(&props, LIMITS + 236); // maxComputeWorkGroupSize[0]
        if max_invocations < 256 || max_local_x < 256 || shared_mem < 2 * 32 * 64 * 4 {
            bail!(
                "{dev_name}: compute limits too small for the GEMM kernel \
                 (workgroup of {max_invocations} invocations, {shared_mem} bytes shared; 256 and 16384 needed)"
            );
        }
        if (max_push as usize) < std::mem::size_of::<Push>() || max_storage < 3 {
            bail!("{dev_name}: push constant or storage buffer limits below the Vulkan minimum");
        }
        // maxMemoryAllocationSize is a 1.1 property; the specification's floor (1 GiB) otherwise.
        let mut alloc_limit = 1u64 << 30;
        if v11 && dev_api >= API_1_1 {
            // VkPhysicalDeviceProperties2 (840 bytes) chained to
            // VkPhysicalDeviceMaintenance3Properties (32: maxMemoryAllocationSize @24).
            let mut m3 = [0u64; 4];
            let mut p2 = [0u64; 105];
            m3[0] = ST_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES as u64;
            p2[0] = ST_PHYSICAL_DEVICE_PROPERTIES_2 as u64;
            p2[1] = m3.as_mut_ptr() as u64;
            if unsafe { vk.vkGetPhysicalDeviceProperties2(pd, p2.as_mut_ptr() as *mut u8) }.is_some() && m3[3] > 0 {
                alloc_limit = m3[3];
            }
        }

        // VkPhysicalDeviceMemoryProperties: memoryTypeCount @0, memoryTypes[32] (flags, heap) @4.
        let mut mp = vec![0u8; 1024];
        unsafe { vk.vkGetPhysicalDeviceMemoryProperties(pd, mp.as_mut_ptr()) };
        let n_types = (rd32(&mp, 0) as usize).min(32);
        let mem_types: Vec<(u32, u32)> = (0..n_types).map(|i| (rd32(&mp, 4 + 8 * i), rd32(&mp, 8 + 8 * i))).collect();

        // The first queue family that does compute (VkQueueFamilyProperties: 24 bytes, flags @0).
        let mut n_fam = 0u32;
        unsafe { vk.vkGetPhysicalDeviceQueueFamilyProperties(pd, &mut n_fam, std::ptr::null_mut()) };
        let mut fams = vec![0u8; 24 * n_fam as usize];
        unsafe { vk.vkGetPhysicalDeviceQueueFamilyProperties(pd, &mut n_fam, fams.as_mut_ptr()) };
        let family = (0..n_fam as usize)
            .find(|&i| rd32(&fams, 24 * i) & QUEUE_COMPUTE != 0 && rd32(&fams, 24 * i + 4) > 0)
            .ok_or_else(|| anyhow!("{dev_name}: no compute queue"))? as u32;

        // VK_EXT_memory_budget where the device has it (VkExtensionProperties: 260 bytes, the
        // name first). It builds on the 1.1 properties2 query.
        let mut budget = false;
        if v11 && dev_api >= API_1_1 {
            let mut n_ext = 0u32;
            if unsafe { vk.vkEnumerateDeviceExtensionProperties(pd, std::ptr::null(), &mut n_ext, std::ptr::null_mut()) } == 0 {
                let mut exts = vec![0u8; 260 * n_ext as usize];
                let rc = unsafe { vk.vkEnumerateDeviceExtensionProperties(pd, std::ptr::null(), &mut n_ext, exts.as_mut_ptr()) };
                if rc >= 0 {
                    budget = exts
                        .chunks(260)
                        .take(n_ext as usize)
                        .any(|e| CStr::from_bytes_until_nul(e).is_ok_and(|s| s == c"VK_EXT_memory_budget"));
                }
            }
        }

        let priority = 1f32;
        let qinfo = DeviceQueueCreateInfo {
            s_type: ST_DEVICE_QUEUE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_family_index: family,
            queue_count: 1,
            priorities: &priority,
        };
        let ext_names = [c"VK_EXT_memory_budget".as_ptr()];
        let dinfo = DeviceCreateInfo {
            s_type: ST_DEVICE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_info_count: 1,
            queue_infos: &qinfo,
            layer_count: 0,
            layers: std::ptr::null(),
            ext_count: budget as u32,
            exts: if budget { ext_names.as_ptr() } else { std::ptr::null() },
            features: std::ptr::null(),
        };
        let mut dev: Ptr = std::ptr::null_mut();
        check(
            unsafe { vk.vkCreateDevice(pd, &dinfo, std::ptr::null(), &mut dev) },
            "vkCreateDevice",
        )?;
        let mut queue: Ptr = std::ptr::null_mut();
        unsafe { vk.vkGetDeviceQueue(dev, family, 0, &mut queue) };
        core.pd = pd;
        core.dev = dev;
        core.queue = queue;
        core.mem_types = mem_types;
        core.budget = budget;
        // Leave some of the allocation count to whatever else the process does with the device.
        core.max_allocs = max_allocs.saturating_sub(64);
        let core = Arc::new(core);
        let vk = &core.vk;

        // One layout for the four kernels: weights, x, y.
        let bindings: Vec<DescriptorSetLayoutBinding> = (0..3)
            .map(|binding| DescriptorSetLayoutBinding {
                binding,
                descriptor_type: DESCRIPTOR_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: STAGE_COMPUTE,
                immutable_samplers: std::ptr::null(),
            })
            .collect();
        let linfo = DescriptorSetLayoutCreateInfo {
            s_type: ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            binding_count: bindings.len() as u32,
            bindings: bindings.as_ptr(),
        };
        // Built up field by field so that a failure half-way frees what exists (Drop).
        let mut be = VulkanBackend {
            core: core.clone(),
            st: Mutex::new(State {
                cmd: std::ptr::null_mut(),
                pools: Vec::new(),
                sets: Vec::new(),
                xh: Buf::default(),
                yh: Buf::default(),
                xd: Buf::default(),
                yd: Buf::default(),
                up: Buf::default(),
            }),
            set_layout: 0,
            layout: 0,
            modules: [0; 4],
            pipelines: [0; 4],
            cmd_pool: 0,
            staged: staged.unwrap_or(dev_type != DEVICE_TYPE_INTEGRATED && dev_type != DEVICE_TYPE_CPU),
            range_limit: max_range,
            alloc_limit,
            max_groups,
            stats: Stats::default(),
            name: format!("vulkan ({dev_name})"),
        };
        check(
            unsafe { vk.vkCreateDescriptorSetLayout(dev, &linfo, std::ptr::null(), &mut be.set_layout) },
            "vkCreateDescriptorSetLayout",
        )?;
        let push = PushConstantRange {
            stage_flags: STAGE_COMPUTE,
            offset: 0,
            size: std::mem::size_of::<Push>() as u32,
        };
        let pinfo = PipelineLayoutCreateInfo {
            s_type: ST_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            set_layouts: &be.set_layout,
            push_constant_range_count: 1,
            push_constant_ranges: &push,
        };
        check(
            unsafe { vk.vkCreatePipelineLayout(dev, &pinfo, std::ptr::null(), &mut be.layout) },
            "vkCreatePipelineLayout",
        )?;
        for (i, bytes) in SPIRV.iter().enumerate() {
            // SPIR-V is a stream of little-endian words.
            let code: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
            let minfo = ShaderModuleCreateInfo {
                s_type: ST_SHADER_MODULE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: 0,
                code_size: code.len() * 4,
                code: code.as_ptr(),
            };
            check(
                unsafe { vk.vkCreateShaderModule(dev, &minfo, std::ptr::null(), &mut be.modules[i]) },
                "vkCreateShaderModule",
            )?;
            let cinfo = ComputePipelineCreateInfo {
                s_type: ST_COMPUTE_PIPELINE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: 0,
                stage: PipelineShaderStageCreateInfo {
                    s_type: ST_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: 0,
                    stage: STAGE_COMPUTE,
                    module: be.modules[i],
                    name: c"main".as_ptr(),
                    specialization_info: std::ptr::null(),
                },
                layout: be.layout,
                base_pipeline: 0,
                base_pipeline_index: -1,
            };
            check(
                unsafe { vk.vkCreateComputePipelines(dev, 0, 1, &cinfo, std::ptr::null(), &mut be.pipelines[i]) },
                "vkCreateComputePipelines",
            )?;
        }
        let cpinfo = CommandPoolCreateInfo {
            s_type: ST_COMMAND_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: COMMAND_POOL_RESET_COMMAND_BUFFER,
            queue_family_index: family,
        };
        check(
            unsafe { vk.vkCreateCommandPool(dev, &cpinfo, std::ptr::null(), &mut be.cmd_pool) },
            "vkCreateCommandPool",
        )?;
        let cbinfo = CommandBufferAllocateInfo {
            s_type: ST_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            pool: be.cmd_pool,
            level: 0,
            count: 1,
        };
        let mut cmd: Ptr = std::ptr::null_mut();
        check(
            unsafe { vk.vkAllocateCommandBuffers(dev, &cbinfo, &mut cmd) },
            "vkAllocateCommandBuffers",
        )?;
        be.st.get_mut().unwrap().cmd = cmd;
        Ok(be)
    }

    /// Starts recording the command buffer (which resets it).
    fn begin(&self, st: &State) -> bool {
        let info = CommandBufferBeginInfo {
            s_type: ST_COMMAND_BUFFER_BEGIN_INFO,
            p_next: std::ptr::null(),
            flags: COMMAND_BUFFER_ONE_TIME_SUBMIT,
            inheritance: std::ptr::null(),
        };
        unsafe { self.core.vk.vkBeginCommandBuffer(st.cmd, &info) == 0 }
    }

    /// Ends the command buffer, submits it and waits for it to finish.
    fn run(&self, st: &State) -> bool {
        let vk = &self.core.vk;
        let submit = SubmitInfo {
            s_type: ST_SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            wait_semaphores: std::ptr::null(),
            wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            command_buffers: &st.cmd,
            signal_semaphore_count: 0,
            signal_semaphores: std::ptr::null(),
        };
        unsafe {
            vk.vkEndCommandBuffer(st.cmd) == 0
                && vk.vkQueueSubmit(self.core.queue, 1, &submit, 0) == 0
                && vk.vkQueueWaitIdle(self.core.queue) == 0
        }
    }

    /// Orders the commands after it behind the writes of the commands before it.
    fn barrier(&self, st: &State, from: (u32, u32), to: (u32, u32)) {
        let b = MemoryBarrier {
            s_type: ST_MEMORY_BARRIER,
            p_next: std::ptr::null(),
            src_access: from.1,
            dst_access: to.1,
        };
        unsafe {
            self.core
                .vk
                .vkCmdPipelineBarrier(st.cmd, from.0, to.0, 0, 1, &b, 0, std::ptr::null(), 0, std::ptr::null())
        };
    }

    /// Copies `parts` (destination offset, bytes) into `buf`: straight into its mapping, or
    /// through the staging buffer when it is device-local.
    fn fill(&self, st: &mut State, buf: &Buf, parts: &[(u64, &[u8])]) -> bool {
        if !buf.ptr.is_null() {
            for (off, bytes) in parts {
                // SAFETY: the parts lie within the buffer (the caller sized it for them).
                unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.ptr.add(*off as usize), bytes.len()) };
            }
            return true;
        }
        if st.up.buffer == 0 {
            let size = UPLOAD_STAGE.min(self.alloc_limit);
            match self.core.create_buf(size, USAGE_TRANSFER_SRC, Place::Host, true, None) {
                Some(b) => st.up = b,
                None => return false,
            }
        }
        let cap = st.up.size as usize;
        for (off, bytes) in parts {
            for (i, piece) in bytes.chunks(cap).enumerate() {
                // SAFETY: a piece is at most the staging buffer's size.
                unsafe { std::ptr::copy_nonoverlapping(piece.as_ptr(), st.up.ptr, piece.len()) };
                let region = BufferCopy {
                    src_offset: 0,
                    dst_offset: off + (i * cap) as u64,
                    size: piece.len() as u64,
                };
                if !self.begin(st) {
                    return false;
                }
                unsafe { self.core.vk.vkCmdCopyBuffer(st.cmd, st.up.buffer, buf.buffer, 1, &region) };
                if !self.run(st) {
                    return false;
                }
            }
        }
        true
    }

    /// Device copy of one weight matrix: None for GGUF weights and shapes the kernel does not
    /// take, or when the device has no room.
    fn upload_weight(&self, w: &WeightRef, n_out: usize, n_in: usize) -> Option<VkWeight> {
        fn bytes<T>(v: &[T]) -> &[u8] {
            // SAFETY: plain numeric data viewed as bytes.
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
        }
        if n_out == 0 || !n_out.is_multiple_of(TILE) || n_in == 0 || !n_in.is_multiple_of(32) {
            return None;
        }
        // The values, scales and offsets with their bytes per row.
        let (fmt, q, s, o, gpr, gk): (usize, &[u8], &[u8], &[u8], usize, usize) = match w {
            WeightRef::Gguf(_) => return None,
            WeightRef::Bf16(v) => (0, bytes(v), &[], &[], 0, 0),
            WeightRef::Q8(m) if (m.n_out, m.n_in) == (n_out, n_in) => (1, bytes(&m.q[..]), bytes(&m.scales[..]), &[], 0, 0),
            WeightRef::Q4(m) if (m.n_out, m.n_in) == (n_out, n_in) => match &m.offsets {
                None => (2, &m.q[..], bytes(&m.scales[..]), &[], 0, 0),
                Some(off) if off.group.is_multiple_of(32) && n_in.is_multiple_of(off.group) => (
                    3,
                    &m.q[..],
                    bytes(&m.scales[..]),
                    bytes(&off.v[..]),
                    n_in / off.group,
                    off.group / 32,
                ),
                Some(_) => return None,
            },
            _ => return None,
        };
        let (q_row, s_row, o_row) = (q.len() / n_out, s.len() / n_out, o.len() / n_out);
        if q.len() != q_row * n_out
            || s.len() != s_row * n_out
            || o.len() != o_row * n_out
            || q_row != [n_in * 2, n_in, n_in / 2, n_in / 2][fmt]
        {
            return None;
        }
        // Whole tiles of rows per chunk, within the binding, allocation and dispatch limits.
        let limit = self.range_limit.min(self.alloc_limit) as usize;
        let per_chunk = ((limit.saturating_sub(16) / (q_row + s_row + o_row)) / TILE * TILE).min(self.max_groups[0] as usize * TILE);
        if per_chunk == 0 {
            return None;
        }
        let mut dw = VkWeight {
            core: self.core.clone(),
            fmt,
            n_out,
            n_in,
            gpr: gpr as u32,
            gk: gk as u32,
            chunks: Vec::with_capacity(n_out.div_ceil(per_chunk)),
        };
        let (usage, place) = if self.staged {
            (USAGE_STORAGE | USAGE_TRANSFER_DST, Place::Device)
        } else {
            (USAGE_STORAGE, Place::Shared)
        };
        let mut st = self.st.lock().unwrap();
        let mut row0 = 0;
        while row0 < n_out {
            let rows = per_chunk.min(n_out - row0);
            // Each part starts on a word (the row counts are multiples of 64, so only odd
            // lengths of 16-bit scales per row could end off one).
            let s_off = (rows * q_row).next_multiple_of(4);
            let o_off = (s_off + rows * s_row).next_multiple_of(4);
            let size = (o_off + rows * o_row).next_multiple_of(16);
            // Dropping `dw` on failure frees the chunks made so far.
            let mut buf = self.core.create_buf(size as u64, usage, place, !self.staged, Some(VRAM_RESERVE))?;
            let ok = self.fill(
                &mut st,
                &buf,
                &[
                    (0, &q[row0 * q_row..(row0 + rows) * q_row]),
                    (s_off as u64, &s[row0 * s_row..(row0 + rows) * s_row]),
                    (o_off as u64, &o[row0 * o_row..(row0 + rows) * o_row]),
                ],
            );
            if !buf.ptr.is_null() {
                unsafe { self.core.vk.vkUnmapMemory(self.core.dev, buf.memory) };
                buf.ptr = std::ptr::null_mut();
            }
            dw.chunks.push(Chunk {
                buf,
                row0,
                rows,
                s_off: s_off as u64,
                o_off: o_off as u64,
            });
            if !ok {
                return None;
            }
            row0 += rows;
        }
        Some(dw)
    }

    /// At least `need` descriptor sets in `st.sets`.
    fn ensure_sets(&self, st: &mut State, need: usize) -> bool {
        let vk = &self.core.vk;
        while st.sets.len() < need {
            let size = DescriptorPoolSize {
                ty: DESCRIPTOR_STORAGE_BUFFER,
                count: 3 * SETS_PER_POOL,
            };
            let info = DescriptorPoolCreateInfo {
                s_type: ST_DESCRIPTOR_POOL_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: 0,
                max_sets: SETS_PER_POOL,
                pool_size_count: 1,
                pool_sizes: &size,
            };
            let mut pool = 0;
            if unsafe { vk.vkCreateDescriptorPool(self.core.dev, &info, std::ptr::null(), &mut pool) } != 0 {
                return false;
            }
            st.pools.push(pool);
            let layouts = [self.set_layout; SETS_PER_POOL as usize];
            let ainfo = DescriptorSetAllocateInfo {
                s_type: ST_DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                pool,
                count: SETS_PER_POOL,
                set_layouts: layouts.as_ptr(),
            };
            let mut sets = [0 as H; SETS_PER_POOL as usize];
            if unsafe { vk.vkAllocateDescriptorSets(self.core.dev, &ainfo, sets.as_mut_ptr()) } != 0 {
                return false;
            }
            st.sets.extend_from_slice(&sets);
        }
        true
    }

    /// Staging buffers of at least `x_bytes` and `y_bytes` (grown with room to spare, within
    /// the allocation limit).
    fn ensure_staging(&self, st: &mut State, x_bytes: u64, y_bytes: u64) -> bool {
        let all = USAGE_STORAGE | USAGE_TRANSFER_SRC | USAGE_TRANSFER_DST;
        let host = if self.staged { Place::Host } else { Place::Shared };
        let grow = |b: &mut Buf, need: u64, place: Place, map: bool| -> bool {
            if b.size >= need {
                return true;
            }
            self.core.destroy_buf(b);
            let cap = (need * 2).min(self.alloc_limit).max(need);
            match self
                .core
                .create_buf(cap, all, place, map, None)
                .or_else(|| self.core.create_buf(need, all, place, map, None))
            {
                Some(n) => {
                    *b = n;
                    true
                }
                None => false,
            }
        };
        grow(&mut st.xh, x_bytes, host, true)
            && grow(&mut st.yh, y_bytes, host, true)
            && (!self.staged || (grow(&mut st.xd, x_bytes, Place::Device, false) && grow(&mut st.yd, y_bytes, Place::Device, false)))
    }

    /// One submission: the jobs' matmuls over tokens `x` (`[t, n_in]`), results into the first
    /// `t` rows of each job's `y` from row `row` on.
    fn pass(&self, st: &mut State, x: &[f32], t: usize, n_in: usize, jobs: &mut [Job], row: usize) -> bool {
        let vk = &self.core.vk;
        let t_pad = t.div_ceil(TILE) * TILE;
        let x_bytes = t_pad * n_in * 4;
        let y_bytes: usize = jobs.iter().map(|j| t_pad * j.n_out * 4).sum();
        let n_dispatch: usize = jobs.iter().map(|j| j.dev.downcast_ref::<VkWeight>().unwrap().chunks.len()).sum();
        if !self.ensure_staging(st, x_bytes as u64, y_bytes as u64) || !self.ensure_sets(st, n_dispatch) {
            return false;
        }
        let t0 = Instant::now();
        // SAFETY: the staging buffers hold at least x_bytes / y_bytes and are mapped.
        unsafe {
            std::ptr::copy_nonoverlapping(x.as_ptr() as *const u8, st.xh.ptr, t * n_in * 4);
            std::ptr::write_bytes(st.xh.ptr.add(t * n_in * 4), 0, (t_pad - t) * n_in * 4);
        }
        let (xb, yb) = if self.staged {
            (st.xd.buffer, st.yd.buffer)
        } else {
            (st.xh.buffer, st.yh.buffer)
        };

        // One descriptor set per dispatch (a job's weight chunk), written before recording.
        let mut infos = Vec::with_capacity(3 * n_dispatch);
        let mut y_off = 0u64;
        for j in jobs.iter() {
            let dw = j.dev.downcast_ref::<VkWeight>().unwrap();
            let y_len = (t_pad * j.n_out * 4) as u64;
            for c in &dw.chunks {
                infos.push(DescriptorBufferInfo {
                    buffer: c.buf.buffer,
                    offset: 0,
                    range: c.buf.size,
                });
                infos.push(DescriptorBufferInfo {
                    buffer: xb,
                    offset: 0,
                    range: x_bytes as u64,
                });
                infos.push(DescriptorBufferInfo {
                    buffer: yb,
                    offset: y_off,
                    range: y_len,
                });
            }
            y_off += y_len;
        }
        let writes: Vec<WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| WriteDescriptorSet {
                s_type: ST_WRITE_DESCRIPTOR_SET,
                p_next: std::ptr::null(),
                dst_set: st.sets[i / 3],
                dst_binding: (i % 3) as u32,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: DESCRIPTOR_STORAGE_BUFFER,
                image_info: std::ptr::null(),
                buffer_info: info,
                texel_buffer_view: std::ptr::null(),
            })
            .collect();
        unsafe { vk.vkUpdateDescriptorSets(self.core.dev, writes.len() as u32, writes.as_ptr(), 0, std::ptr::null()) };

        if !self.begin(st) {
            return false;
        }
        if self.staged {
            let region = BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: x_bytes as u64,
            };
            unsafe { vk.vkCmdCopyBuffer(st.cmd, st.xh.buffer, st.xd.buffer, 1, &region) };
            self.barrier(
                st,
                (PIPELINE_STAGE_TRANSFER, ACCESS_TRANSFER_WRITE),
                (PIPELINE_STAGE_COMPUTE, ACCESS_SHADER_READ),
            );
        }
        let mut set = 0;
        let mut regions = Vec::with_capacity(jobs.len());
        let mut y_off = 0u64;
        for j in jobs.iter() {
            let dw = j.dev.downcast_ref::<VkWeight>().unwrap();
            unsafe { vk.vkCmdBindPipeline(st.cmd, BIND_POINT_COMPUTE, self.pipelines[dw.fmt]) };
            for c in &dw.chunks {
                let push = Push {
                    n_out: dw.n_out as u32,
                    n_in: n_in as u32,
                    row0: c.row0 as u32,
                    s_base: (c.s_off / 4) as u32,
                    o_base: (c.o_off / 4) as u32,
                    gpr: dw.gpr,
                    gk: dw.gk,
                };
                unsafe {
                    vk.vkCmdBindDescriptorSets(st.cmd, BIND_POINT_COMPUTE, self.layout, 0, 1, &st.sets[set], 0, std::ptr::null());
                    vk.vkCmdPushConstants(
                        st.cmd,
                        self.layout,
                        STAGE_COMPUTE,
                        0,
                        std::mem::size_of::<Push>() as u32,
                        &push as *const Push as *const c_void,
                    );
                    vk.vkCmdDispatch(st.cmd, (c.rows / TILE) as u32, (t_pad / TILE) as u32, 1);
                }
                set += 1;
            }
            // Only the real tokens come back.
            regions.push(BufferCopy {
                src_offset: y_off,
                dst_offset: y_off,
                size: (t * j.n_out * 4) as u64,
            });
            y_off += (t_pad * j.n_out * 4) as u64;
        }
        if self.staged {
            self.barrier(
                st,
                (PIPELINE_STAGE_COMPUTE, ACCESS_SHADER_WRITE),
                (PIPELINE_STAGE_TRANSFER, ACCESS_TRANSFER_READ),
            );
            unsafe { vk.vkCmdCopyBuffer(st.cmd, st.yd.buffer, st.yh.buffer, regions.len() as u32, regions.as_ptr()) };
        }
        let t1 = Instant::now();
        if !self.run(st) {
            return false;
        }
        let t2 = Instant::now();
        for (j, r) in jobs.iter_mut().zip(&regions) {
            let dst = &mut j.y[row * j.n_out..(row + t) * j.n_out];
            // SAFETY: the region lies within the mapped result buffer.
            unsafe { std::ptr::copy_nonoverlapping(st.yh.ptr.add(r.src_offset as usize), dst.as_mut_ptr() as *mut u8, dst.len() * 4) };
        }
        self.stats.calls.fetch_add(1, Ordering::Relaxed);
        self.stats.copy_in_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
        self.stats.gpu_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
        self.stats.copy_out_ns.fetch_add(t2.elapsed().as_nanos() as u64, Ordering::Relaxed);
        true
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn stats(&self) -> Option<String> {
        let n = self.stats.calls.load(Ordering::Relaxed);
        if n == 0 {
            return None;
        }
        let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e6;
        Some(format!(
            "{n} submissions: copy-in {:.1} ms, gpu {:.1} ms, copy-out {:.1} ms",
            ms(&self.stats.copy_in_ns),
            ms(&self.stats.gpu_ns),
            ms(&self.stats.copy_out_ns)
        ))
    }

    fn wants(&self, n_out: usize, n_in: usize, t: usize) -> bool {
        let min_flops = if self.staged { MIN_FLOPS_STAGED } else { MIN_FLOPS_DIRECT };
        t >= MIN_T && n_out.is_multiple_of(TILE) && n_in.is_multiple_of(32) && 2.0 * (t * n_out * n_in) as f64 >= min_flops
    }

    fn upload(&self, w: &WeightRef, n_out: usize, n_in: usize) -> Option<DeviceWeight> {
        Some(Box::new(self.upload_weight(w, n_out, n_in)?))
    }

    fn matmul(&self, x: &[f32], t: usize, n_in: usize, jobs: &mut [Job]) -> bool {
        if jobs.is_empty() || t == 0 || x.len() < t * n_in {
            return false;
        }
        let (mut widest, mut total) = (0, 0);
        for j in jobs.iter() {
            match j.dev.downcast_ref::<VkWeight>() {
                Some(dw) if dw.n_out == j.n_out && dw.n_in == n_in && j.y.len() >= t * j.n_out && Arc::ptr_eq(&dw.core, &self.core) => {}
                _ => return false,
            }
            widest = widest.max(j.n_out);
            total += j.n_out;
        }
        // Tokens per submission: x and each job's y within a binding, x and all the y together
        // (one buffer each) within an allocation.
        let (range, alloc) = (self.range_limit as usize / 4, self.alloc_limit as usize / 4);
        let step = (range / n_in.max(widest))
            .min(alloc / n_in.max(total))
            .min(self.max_groups[1] as usize * TILE)
            / TILE
            * TILE;
        if step == 0 {
            return false;
        }
        let mut st = self.st.lock().unwrap();
        let mut row = 0;
        while row < t {
            let n = step.min(t - row);
            if !self.pass(&mut st, &x[row * n_in..(row + n) * n_in], n, n_in, jobs, row) {
                return false;
            }
            row += n;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels;
    use crate::quant::{Q4Matrix, QMatrix, dequantize_rows_q4, from_mlx_q4, mlx_dequant_row, quantize_bf16, quantize_q4};

    /// The backend in each memory mode, or nothing without a Vulkan loader or device. With one,
    /// a backend that cannot start fails the tests (it would otherwise never be tested), except
    /// on a device whose compute limits are below the kernel's.
    fn backends() -> Vec<VulkanBackend> {
        let plat = crate::platform::detect();
        let Some(g) = plat.gpus.iter().find(|g| g.kind == crate::platform::GpuKind::Vulkan) else {
            return Vec::new();
        };
        [true, false]
            .into_iter()
            .filter_map(|staged| match VulkanBackend::with_mode(g.index, Some(staged)) {
                Ok(b) => Some(b),
                Err(e) if e.to_string().contains("limits too small") => None,
                Err(e) => panic!("{e}"),
            })
            .collect()
    }

    fn rnd(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
    fn bf16(n: usize, seed: &mut u64) -> Vec<u16> {
        (0..n).map(|_| (rnd(seed).to_bits() >> 16) as u16).collect()
    }
    /// Token rows of very different magnitudes.
    fn input(t: usize, n: usize, seed: &mut u64) -> Vec<f32> {
        (0..t * n).map(|i| rnd(seed) * [1e-3, 1.0, 300.0, 2e5][(i / n) % 4]).collect()
    }
    /// Worst error of any row relative to that row's largest reference value.
    fn row_err(y: &[f32], r: &[f32], n: usize) -> f32 {
        assert_eq!(y.len(), r.len());
        y.chunks(n)
            .zip(r.chunks(n))
            .map(|(a, b)| {
                let m = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
                a.iter().zip(b).fold(0f32, |e, (p, q)| e.max((p - q).abs())) / m
            })
            .fold(0f32, f32::max)
    }
    /// `[t, n_out]` product with `[n_out, n_in]` f32 weights, exactly (f64).
    fn gemm_ref(w: &[f32], n_out: usize, n_in: usize, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = vec![0f32; t * n_out];
        for i in 0..t {
            for o in 0..n_out {
                y[i * n_out + o] = (0..n_in).map(|k| w[o * n_in + k] as f64 * x[i * n_in + k] as f64).sum::<f64>() as f32;
            }
        }
        y
    }
    /// The same product the way the CPU prefill GEMM and the shader form it: one f32 accumulator
    /// per output, one fused multiply-add per input in order.
    fn gemm_fma(w: &[f32], n_out: usize, n_in: usize, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = vec![0f32; t * n_out];
        for i in 0..t {
            for o in 0..n_out {
                y[i * n_out + o] = (0..n_in).fold(0f32, |a, k| w[o * n_in + k].mul_add(x[i * n_in + k], a));
            }
        }
        y
    }
    fn dense_bf(w: &[u16]) -> Vec<f32> {
        w.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()
    }
    fn dense_q8(m: &QMatrix) -> Vec<f32> {
        let mut f = vec![0f32; m.n_out * m.n_in];
        for (i, o) in f.iter_mut().enumerate() {
            let (r, k) = (i / m.n_in, i % m.n_in);
            *o = m.q[i] as f32 * m.scales[r * (m.n_in / 32) + k / 32];
        }
        f
    }
    fn dense_q4(m: &Q4Matrix) -> Vec<f32> {
        let mut f = vec![0f32; m.n_out * m.n_in];
        dequantize_rows_q4(m, 0, m.n_out, &mut f);
        f
    }
    fn wref<'a>(w: &'a WeightRef<'a>) -> WeightRef<'a> {
        match w {
            WeightRef::Bf16(v) => WeightRef::Bf16(v),
            WeightRef::Q8(m) => WeightRef::Q8(m),
            WeightRef::Q4(m) => WeightRef::Q4(m),
            WeightRef::Gguf(m) => WeightRef::Gguf(m),
        }
    }
    fn run_matmul(be: &VulkanBackend, w: &WeightRef, n_out: usize, n_in: usize, x: &[f32], t: usize) -> Vec<f32> {
        let dev = be.upload(w, n_out, n_in).expect("upload");
        let mut y = vec![f32::NAN; t * n_out];
        let mut jobs = [Job {
            w: wref(w),
            dev: &dev,
            n_out,
            y: &mut y,
        }];
        assert!(be.matmul(x, t, n_in, &mut jobs), "matmul declined");
        y
    }
    /// An MLX-style affine matrix and its values straight from `w = scale * q + bias`.
    fn affine(n_out: usize, n_in: usize, group: usize, seed: &mut u64) -> (Q4Matrix, Vec<f32>) {
        let gpr = n_in / group;
        let words: Vec<u32> = (0..n_out * n_in / 8)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (*seed >> 32) as u32
            })
            .collect();
        let to = |v: f32| half::bf16::from_f32(v).to_bits();
        let scales: Vec<u16> = (0..n_out * gpr).map(|_| to(0.002 + rnd(seed).abs() * 0.02)).collect();
        let biases: Vec<u16> = (0..n_out * gpr).map(|_| to(rnd(seed))).collect();
        let m = from_mlx_q4(&words, &scales, &biases, n_out, n_in, group);
        let mut f = vec![0f32; n_out * n_in];
        for r in 0..n_out {
            let wr = &words[r * n_in / 8..(r + 1) * n_in / 8];
            mlx_dequant_row(
                wr,
                &scales[r * gpr..(r + 1) * gpr],
                &biases[r * gpr..(r + 1) * gpr],
                4,
                group,
                &mut f[r * n_in..(r + 1) * n_in],
            );
        }
        (m, f)
    }
    /// Counts the outputs that are not bit-identical.
    fn mismatches(a: &[f32], b: &[f32]) -> usize {
        a.iter().zip(b).filter(|(p, q)| p.to_bits() != q.to_bits()).count()
    }

    /// Each format against the exact product of its dequantized weights, against the product
    /// in the kernel's order of operations, and against the CPU kernels.
    #[test]
    fn matmul_matches_cpu_and_reference() {
        for be in backends() {
            // 37 blocks per row: the 16-bit Q4 scales of a row end off a word.
            for (n_out, n_in) in [(128, 512), (64, 37 * 32)] {
                let mut seed = 1;
                let w = bf16(n_out * n_in, &mut seed);
                let (q8, q4) = (quantize_bf16(&w, n_out, n_in), quantize_q4(&w, n_out, n_in));
                let formats = [
                    (WeightRef::Bf16(&w), dense_bf(&w)),
                    (WeightRef::Q8(&q8), dense_q8(&q8)),
                    (WeightRef::Q4(&q4), dense_q4(&q4)),
                ];
                // Whole tiles, a partial tile, and fewer tokens than one tile.
                for t in [64, 150, 37, 1] {
                    let x = input(t, n_in, &mut seed);
                    for (w, dense) in &formats {
                        let y = run_matmul(&be, w, n_out, n_in, &x, t);
                        let e = row_err(&y, &gemm_ref(dense, n_out, n_in, &x, t), n_out);
                        assert!(e < 1e-4, "{} t {t}: error {e} against the reference", be.name);
                        let e = row_err(&y, &gemm_fma(dense, n_out, n_in, &x, t), n_out);
                        assert!(e < 1e-5, "{} t {t}: error {e} against the fma-ordered reference", be.name);
                        // Below 32 tokens the CPU's quantized kernels are the int8 decode ones,
                        // approximate by design: the references above are what binds there.
                        if t < 32 {
                            continue;
                        }
                        let mut c = vec![0f32; t * n_out];
                        match w {
                            WeightRef::Bf16(v) => kernels::matmul(v, n_out, n_in, &x, t, &mut c),
                            WeightRef::Q8(m) => kernels::matmul_quant_multi(&mut [(*m, &mut c[..])], &x, t),
                            WeightRef::Q4(m) => kernels::matmul_quant_multi(&mut [(*m, &mut c[..])], &x, t),
                            WeightRef::Gguf(_) => unreachable!(),
                        }
                        let e = row_err(&y, &c, n_out);
                        assert!(e < 1e-5, "{} t {t}: error {e} against the CPU kernels", be.name);
                    }
                }
            }
        }
    }

    /// Whether this driver's results are bit-identical to the fma-ordered reference (the CPU
    /// prefill GEMM's arithmetic). Not asserted: Vulkan lets a driver compile `fma` unfused.
    /// `cargo test --lib vulkan::tests::bit_identity -- --nocapture` prints what this one does.
    #[test]
    fn bit_identity_report() {
        for be in backends() {
            let (n_out, n_in, t) = (128, 1024, 150);
            let mut seed = 7;
            let w = bf16(n_out * n_in, &mut seed);
            let (q8, q4) = (quantize_bf16(&w, n_out, n_in), quantize_q4(&w, n_out, n_in));
            let (qa, _) = affine(n_out, n_in, 64, &mut seed);
            let x = input(t, n_in, &mut seed);
            for (name, w, dense) in [
                ("bf16", WeightRef::Bf16(&w), dense_bf(&w)),
                ("q8", WeightRef::Q8(&q8), dense_q8(&q8)),
                ("q4", WeightRef::Q4(&q4), dense_q4(&q4)),
                ("affine q4", WeightRef::Q4(&qa), dense_q4(&qa)),
            ] {
                let y = run_matmul(&be, &w, n_out, n_in, &x, t);
                let r = gemm_fma(&dense, n_out, n_in, &x, t);
                println!(
                    "{} staged {} {name}: {} of {} outputs differ from the fma-ordered reference (worst row error {:e})",
                    be.name,
                    be.staged,
                    mismatches(&y, &r),
                    y.len(),
                    row_err(&y, &r, n_out)
                );
            }
        }
    }

    /// Several outputs per input in one submission, of different widths and formats, then a
    /// bigger batch (the staging buffers grow) and a smaller one again.
    #[test]
    fn matmul_batches_and_staging_growth() {
        for be in backends() {
            let n_in = 256;
            let mut seed = 2;
            let (a, b) = (bf16(64 * n_in, &mut seed), bf16(192 * n_in, &mut seed));
            let q4 = quantize_q4(&b, 192, n_in);
            let (wa, wb) = (WeightRef::Bf16(&a), WeightRef::Q4(&q4));
            let (da, db) = (be.upload(&wa, 64, n_in).unwrap(), be.upload(&wb, 192, n_in).unwrap());
            let (ra, rb) = (dense_bf(&a), dense_q4(&q4));
            for t in [40, 700, 40] {
                let x = input(t, n_in, &mut seed);
                let (mut ya, mut yb) = (vec![0f32; t * 64], vec![0f32; t * 192]);
                let mut jobs = [
                    Job {
                        w: wref(&wa),
                        dev: &da,
                        n_out: 64,
                        y: &mut ya,
                    },
                    Job {
                        w: wref(&wb),
                        dev: &db,
                        n_out: 192,
                        y: &mut yb,
                    },
                ];
                assert!(be.matmul(&x, t, n_in, &mut jobs));
                assert!(row_err(&ya, &gemm_ref(&ra, 64, n_in, &x, t), 64) < 1e-4);
                assert!(row_err(&yb, &gemm_ref(&rb, 192, n_in, &x, t), 192) < 1e-4);
            }
        }
    }

    #[test]
    fn affine_q4_matmul_matches_mlx_definition() {
        for be in backends() {
            let (n_out, n_in) = (128, 512);
            let mut seed = 3;
            // MLX's group of 64, and groups of one block and of the whole row.
            for group in [64, 32, 512] {
                let (m, truth) = affine(n_out, n_in, group, &mut seed);
                assert!(m.offsets.is_some());
                for t in [64, 150, 37] {
                    let x = input(t, n_in, &mut seed);
                    let y = run_matmul(&be, &WeightRef::Q4(&m), n_out, n_in, &x, t);
                    // The offset and the f16 scale are rounded from the definition's values.
                    let e = row_err(&y, &gemm_ref(&dense_q4(&m), n_out, n_in, &x, t), n_out);
                    assert!(e < 1e-4, "group {group} t {t}: error {e} against the dequantized matrix");
                    let e = row_err(&y, &gemm_ref(&truth, n_out, n_in, &x, t), n_out);
                    assert!(e < 2e-2, "group {group} t {t}: error {e} against w = scale * q + bias");
                    let mut c = vec![0f32; t * n_out];
                    kernels::matmul_quant_multi(&mut [(&m, &mut c[..])], &x, t);
                    let e = row_err(&y, &c, n_out);
                    assert!(e < 1e-5, "group {group} t {t}: error {e} against the CPU kernels");
                }
            }
        }
    }

    fn qbuf<T: Copy>(v: &[T]) -> crate::quant::QBuf<T> {
        let mut p = crate::quant::PageVec::<T>::zeroed(v.len());
        // SAFETY: `p` holds `v.len()` elements.
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), p.as_mut_ptr(), v.len()) };
        p.into()
    }

    /// The shader's hand-written f16 decoding, on every finite f16 scale (subnormals included):
    /// row r, block b has scale bits `r * 512 + b` and a single nonzero weight.
    #[test]
    fn q4_scales_decode_every_finite_f16() {
        for be in backends() {
            let (n_out, bpr) = (128, 512);
            let n_in = bpr * 32;
            let scales: Vec<u16> = (0..n_out * bpr).map(|i| i as u16).collect();
            // Element 0 of every block is 1 (stored 9), the others 0 (stored 8).
            let mut q = vec![0x88u8; n_out * n_in / 2];
            for blk in q.chunks_mut(16) {
                blk[0] = 0x89;
            }
            let m = Q4Matrix {
                q: qbuf(&q),
                scales: qbuf(&scales),
                n_out,
                n_in,
                offsets: None,
            };
            // Token b reads block b's first element.
            let t = bpr;
            let mut x = vec![0f32; t * n_in];
            for b in 0..bpr {
                x[b * n_in + b * 32] = 1.0;
            }
            let y = run_matmul(&be, &WeightRef::Q4(&m), n_out, n_in, &x, t);
            for r in 0..n_out {
                for b in 0..bpr {
                    let h = (r * bpr + b) as u16;
                    let want = half::f16::from_bits(h).to_f32();
                    let got = y[b * n_out + r];
                    // Compared as values: the sum of the row's products is +0 for a scale of -0.
                    if want.is_finite() {
                        assert_eq!(got, want, "f16 {h:#06x}");
                    }
                }
            }
        }
    }

    /// Weights split over several buffers (a small binding limit) and batches run in several
    /// token ranges (a small allocation limit) give the results of the unsplit path.
    #[test]
    fn chunked_weights_and_token_ranges_match_unsplit() {
        for mut be in backends() {
            let (n_out, n_in, t) = (448, 256, 300);
            let mut seed = 5;
            let w = bf16(n_out * n_in, &mut seed);
            let (q8, q4) = (quantize_bf16(&w, n_out, n_in), quantize_q4(&w, n_out, n_in));
            let (qa, _) = affine(n_out, n_in, 64, &mut seed);
            let x = input(t, n_in, &mut seed);
            let formats = [WeightRef::Bf16(&w), WeightRef::Q8(&q8), WeightRef::Q4(&q4), WeightRef::Q4(&qa)];
            let whole: Vec<Vec<f32>> = formats.iter().map(|w| run_matmul(&be, w, n_out, n_in, &x, t)).collect();
            let (range, alloc) = (be.range_limit, be.alloc_limit);
            for (w, r) in formats.iter().zip(&whole) {
                // Within 40 KiB: 64 rows of bf16 (512 bytes each) per chunk, so 7 chunks; 128
                // rows of Q8 (288 bytes), 256 of Q4 (144 bytes, 152 with offsets).
                be.range_limit = 40 << 10;
                let dev = be.upload(w, n_out, n_in).expect("upload");
                be.range_limit = range;
                let chunks = dev.downcast_ref::<VkWeight>().unwrap().chunks.len();
                assert!(chunks >= 2, "{chunks} chunk(s)");
                if matches!(w, WeightRef::Bf16(_)) {
                    assert_eq!(chunks, 7);
                }
                // x of 64 tokens is 64 KiB, y 112 KiB: one tile of tokens per submission.
                be.alloc_limit = 128 << 10;
                let before = be.stats.calls.load(Ordering::Relaxed);
                let mut y = vec![f32::NAN; t * n_out];
                let mut jobs = [Job {
                    w: wref(w),
                    dev: &dev,
                    n_out,
                    y: &mut y,
                }];
                assert!(be.matmul(&x, t, n_in, &mut jobs));
                assert_eq!(be.stats.calls.load(Ordering::Relaxed) - before, 5, "submissions");
                assert_eq!(mismatches(&y, r), 0, "split results differ");
                be.alloc_limit = alloc;
            }
            // A matrix whose single tile of rows exceeds the limit is declined, as is a batch
            // whose single tile of tokens does.
            be.range_limit = 16 << 10;
            assert!(be.upload(&formats[0], n_out, n_in).is_none());
            be.range_limit = range;
            let dev = be.upload(&formats[0], n_out, n_in).unwrap();
            be.alloc_limit = 16 << 10;
            let mut y = vec![0f32; t * n_out];
            let mut jobs = [Job {
                w: wref(&formats[0]),
                dev: &dev,
                n_out,
                y: &mut y,
            }];
            assert!(!be.matmul(&x, t, n_in, &mut jobs));
        }
    }

    /// Shapes and weights the kernel does not take are declined, never miscomputed.
    #[test]
    fn declines_what_it_cannot_run() {
        for be in backends() {
            let mut seed = 6;
            let w = bf16(64 * 64, &mut seed);
            // Rows not a multiple of 64, inputs not a multiple of 32, a wrong length.
            assert!(be.upload(&WeightRef::Bf16(&w[..63 * 64]), 63, 64).is_none());
            assert!(be.upload(&WeightRef::Bf16(&w[..64 * 48]), 64, 48).is_none());
            assert!(be.upload(&WeightRef::Bf16(&w[..64 * 32]), 64, 64).is_none());
            assert!(!be.wants(64, 64, 15) && !be.wants(63, 4096, 4096) && !be.wants(4096, 48, 4096));
            assert!(be.wants(4096, 4096, 64));
            // A weight of another shape than the job's, and too short an output.
            let dev = be.upload(&WeightRef::Bf16(&w), 64, 64).unwrap();
            let x = input(16, 64, &mut seed);
            let mut y = vec![0f32; 16 * 64];
            fn job<'a>(w: &'a [u16], dev: &'a DeviceWeight, y: &'a mut [f32]) -> Job<'a> {
                Job {
                    w: WeightRef::Bf16(w),
                    dev,
                    n_out: 64,
                    y,
                }
            }
            assert!(!be.matmul(&x, 16, 32, &mut [job(&w, &dev, &mut y)]));
            assert!(!be.matmul(&x, 16, 64, &mut [job(&w, &dev, &mut y[..15 * 64])]));
            assert!(!be.matmul(&x[..15 * 64], 16, 64, &mut [job(&w, &dev, &mut y)]));
            assert!(be.matmul(&x, 16, 64, &mut [job(&w, &dev, &mut y)]));
        }
    }

    /// Dropping weights frees their memory: the allocation count returns to where it was.
    #[test]
    fn weights_free_their_memory() {
        for be in backends() {
            let mut seed = 8;
            let w = bf16(64 * 64, &mut seed);
            let x = input(16, 64, &mut seed);
            let mut y = vec![0f32; 16 * 64];
            // Once through, so the staging buffers exist.
            let dev = be.upload(&WeightRef::Bf16(&w), 64, 64).unwrap();
            assert!(be.matmul(
                &x,
                16,
                64,
                &mut [Job {
                    w: WeightRef::Bf16(&w),
                    dev: &dev,
                    n_out: 64,
                    y: &mut y
                }]
            ));
            drop(dev);
            let base = be.core.allocs.load(Ordering::Relaxed);
            let devs: Vec<_> = (0..50).map(|_| be.upload(&WeightRef::Bf16(&w), 64, 64).unwrap()).collect();
            assert_eq!(be.core.allocs.load(Ordering::Relaxed), base + 50);
            drop(devs);
            assert_eq!(be.core.allocs.load(Ordering::Relaxed), base);
        }
    }

    /// Calls from several threads at once do not mix their inputs or results.
    #[test]
    fn concurrent_calls_do_not_mix() {
        for be in backends() {
            let (n_out, n_in, t) = (128, 256, 96);
            let mut seed = 9;
            let w = bf16(n_out * n_in, &mut seed);
            let dense = dense_bf(&w);
            let dev = be.upload(&WeightRef::Bf16(&w), n_out, n_in).unwrap();
            std::thread::scope(|s| {
                for th in 0..4u64 {
                    let (be, w, dev, dense) = (&be, &w, &dev, &dense);
                    s.spawn(move || {
                        let mut seed = 100 + th;
                        for _ in 0..8 {
                            let x = input(t, n_in, &mut seed);
                            let mut y = vec![0f32; t * n_out];
                            let mut jobs = [Job {
                                w: WeightRef::Bf16(w),
                                dev,
                                n_out,
                                y: &mut y,
                            }];
                            assert!(be.matmul(&x, t, n_in, &mut jobs));
                            assert!(row_err(&y, &gemm_ref(dense, n_out, n_in, &x, t), n_out) < 1e-4);
                        }
                    });
                }
            });
        }
    }
}

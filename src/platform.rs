//! Runtime platform detection: CPU architecture and SIMD features, core count, memory, and GPUs
//! (CUDA, Metal, Vulkan) probed by loading the vendor libraries at run time with `dlsym`, so the
//! binary never links against them and runs everywhere with the CPU path as the fallback.

use std::fmt::Write as _;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuKind {
    Cuda,
    Metal,
    Vulkan,
}

impl GpuKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuKind::Cuda => "cuda",
            GpuKind::Metal => "metal",
            GpuKind::Vulkan => "vulkan",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub kind: GpuKind,
    pub index: usize,
    pub name: String,
    /// Device-local (or unified) memory in bytes when the API reports it.
    pub memory_bytes: Option<u64>,
    pub driver: String,
    pub unified_memory: bool,
}

impl GpuInfo {
    /// A Vulkan implementation that runs on the CPU (llvmpipe, SwiftShader).
    pub fn is_software(&self) -> bool {
        self.kind == GpuKind::Vulkan && self.driver.ends_with("(cpu)")
    }
}

#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub arch: &'static str,
    pub os: &'static str,
    pub features: Vec<&'static str>,
    pub cores: usize,
    pub memory_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Platform {
    pub cpu: CpuInfo,
    pub gpus: Vec<GpuInfo>,
    /// Probes that were attempted but found nothing or failed, with the reason.
    pub notes: Vec<String>,
}

/// Which accelerator to use; `Auto` picks the first usable GPU backend (Metal or CUDA before
/// Vulkan), else the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePref {
    Auto,
    Cpu,
    Gpu(GpuKind),
}

impl DevicePref {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "auto" => DevicePref::Auto,
            "cpu" => DevicePref::Cpu,
            "cuda" => DevicePref::Gpu(GpuKind::Cuda),
            "metal" => DevicePref::Gpu(GpuKind::Metal),
            "vulkan" => DevicePref::Gpu(GpuKind::Vulkan),
            _ => return None,
        })
    }
}

pub fn cpu_features() -> Vec<&'static str> {
    #[allow(unused_mut)] // only the aarch64 and x86 detection below push features
    let mut f = Vec::new();
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected as has;
        f.push("neon");
        if has!("dotprod") {
            f.push("dotprod");
        }
        if has!("i8mm") {
            f.push("i8mm");
        }
        if has!("bf16") {
            f.push("bf16");
        }
        if has!("fp16") {
            f.push("fp16");
        }
        if has!("sve") {
            f.push("sve");
        }
        if has!("sve2") {
            f.push("sve2");
        }
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        use std::arch::is_x86_feature_detected as has;
        if has!("sse4.2") {
            f.push("sse4.2");
        }
        if has!("avx") {
            f.push("avx");
        }
        if has!("avx2") {
            f.push("avx2");
        }
        if has!("fma") {
            f.push("fma");
        }
        if has!("f16c") {
            f.push("f16c");
        }
        if has!("avx512f") {
            f.push("avx512f");
        }
        if has!("avx512bw") {
            f.push("avx512bw");
        }
        if has!("avx512vnni") {
            f.push("avx512vnni");
        }
        if has!("avxvnni") {
            f.push("avxvnni");
        }
        if has!("avx512bf16") {
            f.push("avx512bf16");
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        if cfg!(target_feature = "simd128") {
            f.push("simd128");
        }
        if cfg!(target_feature = "atomics") {
            f.push("atomics");
        }
    }
    f
}

/// Memory oxinfer always leaves to the rest of the system: 20% of the physical memory, at most
/// 2 GB. Loading decisions keep [`available_memory`] above it (see [`memory_headroom`]).
pub fn memory_reserve() -> u64 {
    total_memory().map_or(2 << 30, |t| (t / 5).min(2 << 30))
}

/// Memory the system could give a process now without swapping: Linux's `MemAvailable` (free
/// memory plus reclaimable page cache); elsewhere what the OS lets this app still use (iOS), else
/// the physical memory.
pub fn available_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        return Some(kb * 1024);
    }
    #[allow(unreachable_code)]
    process_available_memory().or_else(total_memory)
}

/// What a load may still take for itself: [`available_memory`] minus [`memory_reserve`]. None
/// when the platform reports neither.
pub fn memory_headroom() -> Option<u64> {
    available_memory().map(|a| a.saturating_sub(memory_reserve()))
}

/// Whether `bytes` more of process memory leaves the system its reserve.
pub fn fits_in_memory(bytes: u64) -> bool {
    memory_headroom().is_none_or(|h| bytes <= h)
}

/// Physical memory in bytes, when the platform reports it.
pub fn total_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = s.lines().find(|l| l.starts_with("MemTotal:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        return Some(kb * 1024);
    }
    #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "gpu"))]
    {
        // sysctlbyname("hw.memsize") from the already-loaded libSystem.
        let lib = dlsym::Library::this().ok()?;
        type F = unsafe extern "C" fn(
            *const std::ffi::c_char,
            *mut std::ffi::c_void,
            *mut usize,
            *mut std::ffi::c_void,
            usize,
        ) -> std::ffi::c_int;
        let f = unsafe { lib.get::<F>(c"sysctlbyname") }.ok()?;
        let mut v: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let rc = unsafe {
            f.call((
                c"hw.memsize".as_ptr(),
                &mut v as *mut u64 as *mut _,
                &mut len,
                std::ptr::null_mut(),
                0,
            ))
        };
        return if rc == 0 { Some(v) } else { None };
    }
    #[allow(unreachable_code)]
    None
}

/// Memory this process may still allocate before the OS ends it, when the platform says: iOS's
/// `os_proc_available_memory` (the app's jetsam limit minus its footprint; raised by the
/// `com.apple.developer.kernel.increased-memory-limit` entitlement). Clean file pages (mapped
/// weights in the page cache) do not count against it.
pub fn process_available_memory() -> Option<u64> {
    #[cfg(all(target_os = "ios", feature = "gpu"))]
    {
        let lib = dlsym::Library::this().ok()?;
        type F = unsafe extern "C" fn() -> usize;
        let f = unsafe { lib.get::<F>(c"os_proc_available_memory") }.ok()?;
        let v = unsafe { f.call(()) } as u64;
        return (v > 0).then_some(v);
    }
    #[allow(unreachable_code)]
    None
}

pub fn detect() -> Platform {
    let cpu = CpuInfo {
        arch: std::env::consts::ARCH,
        os: std::env::consts::OS,
        features: cpu_features(),
        cores: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        memory_bytes: total_memory(),
    };
    #[allow(unused_mut)]
    let mut gpus = Vec::new();
    #[cfg(feature = "gpu")]
    let notes = {
        let mut n = Vec::new();
        gpu::probe_all(&mut gpus, &mut n);
        n
    };
    #[cfg(not(feature = "gpu"))]
    let notes = vec!["gpu feature disabled at build time".to_string()];
    Platform { cpu, gpus, notes }
}

impl Platform {
    pub fn summary(&self) -> String {
        let mut s = format!("{} {}, {} cores", self.cpu.arch, self.cpu.os, self.cpu.cores);
        if !self.cpu.features.is_empty() {
            let _ = write!(s, ", {}", self.cpu.features.join("+"));
        }
        if let Some(m) = self.cpu.memory_bytes {
            let _ = write!(s, ", {:.0} GB RAM", m as f64 / 1e9);
        }
        for g in &self.gpus {
            let _ = write!(s, "; {} #{} {}", g.kind.as_str(), g.index, g.name);
            if let Some(m) = g.memory_bytes {
                let _ = write!(s, " ({:.0} GB{})", m as f64 / 1e9, if g.unified_memory { ", unified" } else { "" });
            }
        }
        s
    }

    pub fn report(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "CPU: {} on {}, {} logical cores", self.cpu.arch, self.cpu.os, self.cpu.cores);
        let _ = writeln!(
            s,
            "  features: {}",
            if self.cpu.features.is_empty() {
                "(baseline)".to_string()
            } else {
                self.cpu.features.join(" ")
            }
        );
        #[cfg(target_arch = "x86_64")]
        let _ = writeln!(
            s,
            "  kernels: {} (OXINFER_SIMD=scalar selects the fallbacks)",
            crate::kernels::simd::level_name()
        );
        if let Some(m) = self.cpu.memory_bytes {
            let _ = writeln!(s, "  memory: {:.1} GB", m as f64 / 1e9);
        }
        if self.gpus.is_empty() {
            let _ = writeln!(s, "GPU: none usable");
        }
        for g in &self.gpus {
            let _ = writeln!(s, "GPU ({} #{}): {}", g.kind.as_str(), g.index, g.name);
            if let Some(m) = g.memory_bytes {
                let _ = writeln!(
                    s,
                    "  memory: {:.1} GB{}",
                    m as f64 / 1e9,
                    if g.unified_memory { " (unified with CPU)" } else { "" }
                );
            }
            if !g.driver.is_empty() {
                let _ = writeln!(s, "  driver: {}", g.driver);
            }
        }
        for n in &self.notes {
            let _ = writeln!(s, "  note: {n}");
        }
        s
    }

    /// The devices `auto` tries, in order: the first of each backend (Metal or CUDA before
    /// Vulkan), without Vulkan's software rasterizers (slower than the CPU kernels).
    pub fn auto_order(&self) -> impl Iterator<Item = &GpuInfo> {
        self.gpus.iter().enumerate().filter_map(|(i, g)| {
            let first = !self.gpus[..i].iter().any(|x| x.kind == g.kind);
            (first && !g.is_software()).then_some(g)
        })
    }

    /// Resolves a preference to a concrete GPU, if any.
    pub fn pick(&self, pref: DevicePref) -> Option<&GpuInfo> {
        match pref {
            DevicePref::Cpu => None,
            DevicePref::Auto => self.auto_order().next(),
            DevicePref::Gpu(k) => self.gpus.iter().find(|g| g.kind == k),
        }
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::{GpuInfo, GpuKind};
    use std::ffi::{CStr, c_char, c_int, c_void};

    pub fn probe_all(gpus: &mut Vec<GpuInfo>, notes: &mut Vec<String>) {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        match metal::probe() {
            Ok(Some(g)) => gpus.push(g),
            Ok(None) => notes.push("metal: no device".into()),
            Err(e) => notes.push(format!("metal: {e}")),
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32")))]
        match cuda::probe() {
            Ok(v) if !v.is_empty() => gpus.extend(v),
            Ok(_) => notes.push("cuda: no device".into()),
            Err(e) => notes.push(format!("cuda: {e}")),
        }
        #[cfg(not(target_arch = "wasm32"))]
        match vulkan::probe() {
            // After the vendor backends (which `auto` prefers), and listed even when the same
            // physical GPU is already there through CUDA or Metal: `--device vulkan` picks them.
            Ok(v) if !v.is_empty() => gpus.extend(v),
            Ok(_) => notes.push("vulkan: no device".into()),
            Err(e) => notes.push(format!("vulkan: {e}")),
        }
        let _ = (&gpus, &notes);
    }

    #[allow(dead_code)]
    fn cstr(p: *const c_char) -> String {
        if p.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    /// CUDA driver API via libcuda / nvcuda.dll.
    #[cfg(not(any(target_os = "macos", target_os = "ios", target_arch = "wasm32")))]
    pub mod cuda {
        use super::*;
        dlsym::library! {
            pub struct Cuda {
                unsafe extern "C" {
                    pub fn cuInit(flags: u32) -> c_int;
                    pub fn cuDriverGetVersion(v: *mut c_int) -> c_int;
                    pub fn cuDeviceGetCount(n: *mut c_int) -> c_int;
                    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: c_int) -> c_int;
                    pub fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: c_int) -> c_int;
                    #[optional]
                    pub fn cuDeviceGetAttribute(v: *mut c_int, attr: c_int, dev: c_int) -> c_int;
                }
            }
        }
        pub const CANDIDATES: &[&str] = &["libcuda.so.1", "libcuda.so", "nvcuda.dll"];

        pub fn open() -> Result<Cuda, String> {
            use dlsym::Bindings;
            let mut last = String::new();
            for c in CANDIDATES {
                match unsafe { dlsym::Library::open(c) }
                    .map_err(|e| e.to_string())
                    .and_then(|l| Cuda::load(l).map_err(|e| e.to_string()))
                {
                    Ok(l) => return Ok(l),
                    Err(e) => last = e,
                }
            }
            let _ = &last;
            Err("libcuda not installed".to_string())
        }

        pub fn probe() -> Result<Vec<GpuInfo>, String> {
            let cu = open()?;
            let rc = unsafe { cu.cuInit(0) };
            if rc != 0 {
                return Err(format!("cuInit failed ({rc})"));
            }
            let mut ver = 0;
            unsafe { cu.cuDriverGetVersion(&mut ver) };
            let mut n = 0;
            unsafe { cu.cuDeviceGetCount(&mut n) };
            let mut out = Vec::new();
            for d in 0..n {
                let mut name = [0 as c_char; 256];
                unsafe { cu.cuDeviceGetName(name.as_mut_ptr(), 256, d) };
                let mut mem = 0usize;
                unsafe { cu.cuDeviceTotalMem_v2(&mut mem, d) };
                let mut cc = String::new();
                let (mut major, mut minor) = (0, 0);
                // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / MINOR
                if unsafe { cu.cuDeviceGetAttribute(&mut major, 75, d) }.is_some() {
                    unsafe { cu.cuDeviceGetAttribute(&mut minor, 76, d) };
                    cc = format!(", compute capability {major}.{minor}");
                }
                out.push(GpuInfo {
                    kind: GpuKind::Cuda,
                    index: d as usize,
                    name: cstr(name.as_ptr()),
                    memory_bytes: Some(mem as u64),
                    driver: format!("CUDA driver {}.{}{cc}", ver / 1000, (ver % 1000) / 10),
                    unified_memory: false,
                });
            }
            Ok(out)
        }
    }

    /// Metal via the Objective-C runtime (no bindings, just `objc_msgSend`).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub mod metal {
        use super::*;
        pub type Id = *mut c_void;
        pub type Sel = *mut c_void;

        pub struct Objc {
            pub metal: dlsym::Library,
            pub objc: dlsym::Library,
        }

        impl Objc {
            pub fn open() -> Result<Self, String> {
                let metal =
                    unsafe { dlsym::Library::open("/System/Library/Frameworks/Metal.framework/Metal") }.map_err(|e| e.to_string())?;
                let objc = unsafe { dlsym::Library::open("/usr/lib/libobjc.A.dylib") }.map_err(|e| e.to_string())?;
                Ok(Self { metal, objc })
            }
            pub fn sel(&self, name: &CStr) -> Sel {
                type F = unsafe extern "C" fn(*const c_char) -> Sel;
                let f = unsafe { self.objc.get::<F>(c"sel_registerName") }.expect("sel_registerName");
                unsafe { f.call((name.as_ptr(),)) }
            }
            /// `objc_msgSend` typed for the call at hand. The caller picks the signature.
            pub fn msg_send<T: dlsym::FnPtr>(&self) -> T {
                let s = unsafe { self.objc.get::<T>(c"objc_msgSend") }.expect("objc_msgSend");
                unsafe { s.into_inner() }
            }
            pub fn default_device(&self) -> Id {
                type F = unsafe extern "C" fn() -> Id;
                match unsafe { self.metal.get::<F>(c"MTLCreateSystemDefaultDevice") } {
                    Ok(f) => unsafe { f.call(()) },
                    Err(_) => std::ptr::null_mut(),
                }
            }
            /// Copies an `NSString` into a Rust string (empty for null).
            ///
            /// # Safety
            /// `s` must be null or a valid `NSString` object pointer.
            pub unsafe fn nsstring_to_string(&self, s: Id) -> String {
                if s.is_null() {
                    return String::new();
                }
                let f: unsafe extern "C" fn(Id, Sel) -> *const c_char = self.msg_send();
                cstr(unsafe { f(s, self.sel(c"UTF8String")) })
            }
        }

        pub fn probe() -> Result<Option<GpuInfo>, String> {
            let o = Objc::open()?;
            let dev = o.default_device();
            if dev.is_null() {
                return Ok(None);
            }
            let send_id: unsafe extern "C" fn(Id, Sel) -> Id = o.msg_send();
            let send_u64: unsafe extern "C" fn(Id, Sel) -> u64 = o.msg_send();
            let send_bool: unsafe extern "C" fn(Id, Sel) -> u8 = o.msg_send();
            // SAFETY: `name` returns an NSString owned by the device.
            let name = unsafe { o.nsstring_to_string(send_id(dev, o.sel(c"name"))) };
            let mem = unsafe { send_u64(dev, o.sel(c"recommendedMaxWorkingSetSize")) };
            let unified = unsafe { send_bool(dev, o.sel(c"hasUnifiedMemory")) } != 0;
            Ok(Some(GpuInfo {
                kind: GpuKind::Metal,
                index: 0,
                name,
                memory_bytes: if mem > 0 { Some(mem) } else { None },
                driver: "Metal".to_string(),
                unified_memory: unified,
            }))
        }
    }

    /// Vulkan loader: enumerate physical devices and their device-local memory.
    #[cfg(not(target_arch = "wasm32"))]
    pub mod vulkan {
        use super::*;
        dlsym::library! {
            pub struct Vk {
                unsafe extern "C" {
                    pub fn vkCreateInstance(info: *const c_void, alloc: *const c_void, out: *mut *mut c_void) -> c_int;
                    pub fn vkDestroyInstance(inst: *mut c_void, alloc: *const c_void);
                    pub fn vkEnumeratePhysicalDevices(inst: *mut c_void, count: *mut u32, devs: *mut *mut c_void) -> c_int;
                    pub fn vkGetPhysicalDeviceProperties(dev: *mut c_void, props: *mut u8);
                    pub fn vkGetPhysicalDeviceMemoryProperties(dev: *mut c_void, props: *mut u8);
                }
            }
        }
        pub const CANDIDATES: &[&str] = &[
            "libvulkan.so.1",
            "libvulkan.so",
            "vulkan-1.dll",
            "libvulkan.1.dylib",
            "libvulkan.dylib",
            "libMoltenVK.dylib",
        ];

        #[repr(C)]
        struct InstanceCreateInfo {
            s_type: u32,
            p_next: *const c_void,
            flags: u32,
            app_info: *const c_void,
            layer_count: u32,
            layers: *const *const c_char,
            ext_count: u32,
            exts: *const *const c_char,
        }

        pub fn probe() -> Result<Vec<GpuInfo>, String> {
            use dlsym::Bindings;
            let mut lib = None;
            let mut last = String::new();
            for c in CANDIDATES {
                match unsafe { dlsym::Library::open(c) }
                    .map_err(|e| e.to_string())
                    .and_then(|l| Vk::load(l).map_err(|e| e.to_string()))
                {
                    Ok(l) => {
                        lib = Some(l);
                        break;
                    }
                    Err(e) => last = e,
                }
            }
            let _ = &last;
            let vk = lib.ok_or_else(|| "libvulkan not installed".to_string())?;
            let info = InstanceCreateInfo {
                s_type: 1, // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
                p_next: std::ptr::null(),
                flags: 0,
                app_info: std::ptr::null(),
                layer_count: 0,
                layers: std::ptr::null(),
                ext_count: 0,
                exts: std::ptr::null(),
            };
            let mut inst: *mut c_void = std::ptr::null_mut();
            let rc = unsafe { vk.vkCreateInstance(&info as *const _ as *const c_void, std::ptr::null(), &mut inst) };
            if rc != 0 || inst.is_null() {
                return Err(format!("vkCreateInstance failed ({rc})"));
            }
            let mut count = 0u32;
            unsafe { vk.vkEnumeratePhysicalDevices(inst, &mut count, std::ptr::null_mut()) };
            let mut devs = vec![std::ptr::null_mut::<c_void>(); count as usize];
            unsafe { vk.vkEnumeratePhysicalDevices(inst, &mut count, devs.as_mut_ptr()) };
            let mut out = Vec::new();
            for (i, &d) in devs.iter().enumerate().take(count as usize) {
                // VkPhysicalDeviceProperties: apiVersion u32 @0, driverVersion @4, vendorID @8,
                // deviceID @12, deviceType @16, deviceName[256] @20, ... (we over-allocate).
                let mut props = vec![0u8; 2048];
                unsafe { vk.vkGetPhysicalDeviceProperties(d, props.as_mut_ptr()) };
                let api = u32::from_ne_bytes(props[0..4].try_into().unwrap());
                let dev_type = u32::from_ne_bytes(props[16..20].try_into().unwrap());
                let name = cstr(props[20..].as_ptr() as *const c_char);
                // VkPhysicalDeviceMemoryProperties: memoryTypeCount u32 @0, memoryTypes[32] (8 B each) @4,
                // memoryHeapCount u32 @260, memoryHeaps[16] (size u64, flags u32, pad) @264.
                let mut mp = vec![0u8; 1024];
                unsafe { vk.vkGetPhysicalDeviceMemoryProperties(d, mp.as_mut_ptr()) };
                let heaps = u32::from_ne_bytes(mp[260..264].try_into().unwrap()) as usize;
                let mut local = 0u64;
                for h in 0..heaps.min(16) {
                    let o = 264 + h * 16;
                    let size = u64::from_ne_bytes(mp[o..o + 8].try_into().unwrap());
                    let flags = u32::from_ne_bytes(mp[o + 8..o + 12].try_into().unwrap());
                    if flags & 1 != 0 {
                        local += size;
                    }
                }
                let kind_s = match dev_type {
                    1 => "integrated",
                    2 => "discrete",
                    3 => "virtual",
                    4 => "cpu",
                    _ => "other",
                };
                out.push(GpuInfo {
                    kind: GpuKind::Vulkan,
                    index: i,
                    name,
                    memory_bytes: if local > 0 { Some(local) } else { None },
                    driver: format!("Vulkan {}.{}.{} ({kind_s})", api >> 22, (api >> 12) & 0x3ff, api & 0xfff),
                    unified_memory: dev_type == 1,
                });
            }
            unsafe { vk.vkDestroyInstance(inst, std::ptr::null()) };
            // The order `--device vulkan` and `auto` take them in: `OXINFER_VULKAN_DEVICE=N` (the
            // index listed by `--devices`) first when set, then discrete devices, integrated,
            // virtual, others and software rasterizers; the larger memory first within a type.
            let forced: Option<usize> = std::env::var("OXINFER_VULKAN_DEVICE").ok().and_then(|v| v.parse().ok());
            let rank = |g: &GpuInfo| {
                let ty = ["(discrete)", "(integrated)", "(virtual)", "(other)", "(cpu)"]
                    .iter()
                    .position(|t| g.driver.ends_with(t))
                    .unwrap_or(3);
                (Some(g.index) != forced, ty, std::cmp::Reverse(g.memory_bytes.unwrap_or(0)))
            };
            out.sort_by_key(rank);
            Ok(out)
        }
    }
}

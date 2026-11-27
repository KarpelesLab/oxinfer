//! `W x` of one of ComfyUI's quantized weights as oxinfer computes it, for
//! `tools/comfy_quant_check.py` to compare with comfy_kitchen's dequantized weight:
//!
//! ```text
//! comfy_quant_check FILE.safetensors TENSOR.weight OUT_DIR [ROWS]
//! ```
//!
//! Writes `OUT_DIR/x.bin` (`ROWS` seeded normal input rows, default 8) and `OUT_DIR/y.bin` (the
//! products), float32, and prints the shape.

use anyhow::{Context, Result};
use std::path::Path;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: comfy_quant_check FILE TENSOR.weight OUT_DIR [ROWS]";
    let (file, name, out) = (
        a.first().context(usage)?,
        a.get(1).context(usage)?,
        Path::new(a.get(2).context(usage)?),
    );
    let t: usize = a.get(3).map_or(Ok(8), |s| s.parse()).context(usage)?;
    oxinfer::gpu::init(&oxinfer::platform::detect(), oxinfer::platform::DevicePref::Cpu)?;
    let (n_out, n_in, x, y) = oxinfer::comfy_quant::check_product(Path::new(file), name, t, 7)?;
    std::fs::create_dir_all(out)?;
    let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    std::fs::write(out.join("x.bin"), bytes(&x))?;
    std::fs::write(out.join("y.bin"), bytes(&y))?;
    println!("{name}: [{n_out}, {n_in}], {t} rows");
    Ok(())
}

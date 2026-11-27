# Depth estimation (Depth Pro 2)

Apple's [Depth Pro 2](https://huggingface.co/apple/DepthPro2) predicts metric depth for every pixel
of a photo, at the photo's full resolution, and the focal length when the photo does not carry
one. Each size is its own repository; all share the focal length estimator
[`apple/DepthPro2-focal`](https://huggingface.co/apple/DepthPro2-focal) (23.5M parameters, 94 MB),
which oxinfer loads beside the size by default:

| | backbone | parameters | download |
|---|---|---|---|
| `apple/DepthPro2-S` | DINOv3 ViT-S/16 | 24M | 97 MB |
| `apple/DepthPro2-B` | DINOv3 ViT-B/16 | 97M | 386 MB |
| `apple/DepthPro2-L` | DINOv3 ViT-L/16 | 334M | 1.3 GB |
| `apple/DepthPro2-H` | DINOv3 ViT-H+/16 (SwiGLU) | 872M | 3.5 GB |

The weights are under the Apple Machine Learning Research Model license (research use only).

`src/depth_pro.rs` follows the reference package
([apple-aiml-research/ml-depth-pro-2](https://github.com/apple-aiml-research/ml-depth-pro-2):
`model.py`, `network/encoder.py`, `network/decoder.py`, `network/fov.py`) and timm's DINOv3 ViT:
the ImageNet-normalized image, padded to a multiple of 32 (edges replicated), through the ViT
(class and four register tokens, 2D rotary positions normalized by the longer side, layer scale;
for one image the "cross-frame" blocks keep their input class and register tokens), four blocks'
patch tokens reassembled into a 1/4 to 1/32 pyramid, the DPT decoder and the head giving
focal-normalized inverse depth at full resolution; depth in meters is
`f_px / (width * inverse_depth)`. The focal length comes from `--focal-px`, else the photo's EXIF
35 mm equivalent focal length (as the reference's `load_rgb`), else the estimator: a DINOv3
ViT-S/16 over the image resized with torch's antialiased bilinear filter to a short side of at most
512 (centre-cropped beyond 16:9) and an attention head that regresses both fields of view. EXIF
orientation is applied.

## Use

```sh
oxinfer depth photo.jpg                              # photo-depth.png beside it (S + the estimator)
oxinfer depth photo.jpg --model apple/DepthPro2-L --png16 --raw --out results/
```

`oxinfer depth IMAGE...` writes, like the reference's `depth-pro-2` command, `NAME-depth.png`
(metric inverse depth through matplotlib's turbo color map, from 0 at infinite depth to the nearest
point plus 20%), with `--png16` `NAME-depth16.png` (16-bit gray inverse depth on the same scale:
depth in meters = k / value, k printed), with `--raw` `NAME-depth.f32` (depth in meters, float32,
row-major, infinite where infinitely far), and prints the focal length and the depth range (5th to
95th percentile). Options: `--model`, `--focal SPEC|none`, `--focal-px F`, `--no-exif`, `--out DIR`,
`--device`, `--threads`, `--cache-dir`.

Library:

```rust
use oxinfer::depth_pro::{DepthOptions, DepthPro};
let net = DepthPro::load("apple/DepthPro2-S", &DepthOptions::default())?;
let img = oxinfer::depth_pro::decode_image(&std::fs::read("photo.jpg")?)?;
let d = net.infer_rgb(&img.rgb, img.width, img.height, img.focal_px)?;
let meters = d.depth().unwrap();          // width * height, row-major
println!("{:?} px, fov {:?}", d.focal_px, d.fov_deg);
```

C: `oxi_depth_load` (`focal=none`, `device=cpu`, ...), `oxi_depth_rgb` (depth and inverse depth
buffers, figures as JSON), `oxi_depth_file` (writes the three files), `oxi_depth_info_json`,
`oxi_depth_free` (see `include/oxinfer.h`).

Server: model kind `depth` (detected from the repository's `config.json`; option `focal`),
`POST /v1/images/depth` with `model`, `file` / `image` (multipart, or base64 / a data URL in JSON),
`focal_px`, `exif`, `response_format` (`json`, the default: focal length, fields of view, depth
range, base64 `png` and `png16`, with `include_depth` the base64 float32 depth; `png`; `png16`;
`f32`).

```sh
oxinfer serve --model depth=apple/DepthPro2-S
curl -s localhost:8080/v1/images/depth -F model=depth -F file=@photo.jpg -F response_format=png -o depth.png
```

Not done: video (the reference's buffered and unbuffered multi-frame modes, where the cross-frame
blocks attend across frames).

## Accuracy

Against the reference package (torch 2.15 CPU, float32, timm 1.0) on the repository's
`assets/example.jpg` (as PNG, 1920x1440, and halved to 960x720), with the estimated focal length:

| | inverse depth: max abs difference (its range) | depth: median / 99th percentile relative difference | focal length |
|---|---|---|---|
| S, 960x720, CPU | 1.2e-6 (0 to 0.142) | 1.9e-6 / 6.6e-6 | 781.6365 px, the reference's to 3e-8 |
| S, 960x720, Metal | 4.3e-7 | 4.8e-7 / 3.2e-6 | same |
| B, 960x720 | 1.2e-6 (0 to 0.115) | 8.3e-7 / 2.7e-6 | same |
| L, 960x720 | 5.5e-7 (0 to 0.115) | 7.8e-7 / 3.2e-6 | same |
| S, 1920x1440 | 2.9e-6 (0 to 0.126) | 6.0e-7 / 2.6e-6 | 1566.83 px, same |

(The largest relative depth differences, a few percent, are at the sky's edge, where the inverse
depth is near zero and the depth near infinite.) H was not run: its SwiGLU MLP follows timm's
`SwiGLU` (`fc2(silu(fc1_g x) * fc1_x x)`, the tensor names of the checkpoint), unverified.
`tests` in `src/depth_pro.rs` run a tiny random model end to end (padding, the focal estimator,
given focal lengths, the outputs), check the antialiased resize against torch's and the EXIF
reader; `src/server/tools.rs` tests the endpoint.

## Speed

M5 Max (18 cores), one image, the estimated focal length included (the estimator adds about
0.2 s), weights loaded; measured under `tools/bench/with-lock.sh` on a machine shared with other
builds:

| | 960x720 Metal | 960x720 CPU | 1920x1440 Metal | 1920x1440 CPU | reference (torch CPU), 960x720 / 1920x1440 |
|---|---|---|---|---|---|
| S | 0.50 s | 0.74 s | 1.32 s | 4.6 s | 0.79 s / 3.2 s |
| B | 0.60 s | 1.47 s | 2.05 s | 8.6 s | 2.3 s / - |
| L | 1.08 s | 3.12 s | 4.30 s | 24.5 s | 5.7 s / - |

With Metal the matmuls, convolutions and attention run per op on the GPU (activations copied in
and out for each); at 1920x1440 the decoder's full-resolution convolutions (2.8 M pixels, 32
channels) and those copies take as long as the ViT. `--profile` prints the split.

//! A small ONNX reader and interpreter: the protobuf decoded by hand (no dependency), the graph
//! run node by node in f32. It covers the operators of small convolutional models (`Conv` 1D /
//! 2D with groups, element-wise ops, reductions, nearest `Resize`, `Gemm`, `ArgMax`, and
//! `QuantizeLinear` / `DequantizeLinear`, run as float fake quantization), which is what
//! OCRia-nano's detector, router and recognizers use ([`crate::ocria`]).
//!
//! At load time constant subgraphs (`Constant` nodes, weights through `DequantizeLinear`) are
//! folded, a `QuantizeLinear` feeding one `DequantizeLinear` becomes one fake-quantization step,
//! and `Relu` / fake quantization after a `Conv` or an `Add` run inside it. Convolutions run
//! channels-last (`[N, H, W, C]` in memory while shapes stay `[N, C, H, W]`): pointwise ones as
//! one GEMM over the pixels ([`crate::sd::ops::matmul_f32`]), depthwise ones vectorized across
//! the channels, the others as im2col + GEMM; tensors go back to `NCHW` where an operator needs
//! it, and every graph output is `NCHW`.
//!
//! Quantized graphs are computed in float: an int8 runtime (ONNX Runtime's `QLinearConv`
//! kernels) rounds its accumulations differently, so its results can differ slightly.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;

// ---------------------------------------------------------------------------------------------
// Protobuf.

struct Pb<'a> {
    b: &'a [u8],
    i: usize,
}

enum Val<'a> {
    Varint(u64),
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32(u32),
}

impl<'a> Pb<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, i: 0 }
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *self.b.get(self.i).context("truncated varint")?;
            self.i += 1;
            v |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(v);
            }
        }
        bail!("varint too long")
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.i.checked_add(n).context("truncated field")?;
        let s = self.b.get(self.i..end).context("truncated field")?;
        self.i = end;
        Ok(s)
    }

    /// The next field: its number and value, or None at the end.
    fn next(&mut self) -> Result<Option<(u32, Val<'a>)>> {
        if self.i >= self.b.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let (num, wt) = ((key >> 3) as u32, key & 7);
        let v = match wt {
            0 => Val::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                Val::Fixed64
            }
            2 => {
                let n = usize::try_from(self.varint()?).context("field length")?;
                Val::Bytes(self.take(n)?)
            }
            5 => Val::Fixed32(u32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            _ => bail!("unsupported protobuf wire type {wt}"),
        };
        Ok(Some((num, v)))
    }
}

fn utf8(b: &[u8]) -> Result<String> {
    Ok(std::str::from_utf8(b).context("protobuf string is not UTF-8")?.to_string())
}

/// Varints, packed or one per field.
fn ints(v: &Val, out: &mut Vec<i64>) -> Result<()> {
    match v {
        Val::Varint(x) => out.push(*x as i64),
        Val::Bytes(b) => {
            let mut p = Pb::new(b);
            while p.i < b.len() {
                out.push(p.varint()? as i64);
            }
        }
        _ => bail!("expected varints"),
    }
    Ok(())
}

fn floats(v: &Val, out: &mut Vec<f32>) -> Result<()> {
    match v {
        Val::Fixed32(x) => out.push(f32::from_bits(*x)),
        Val::Bytes(b) => out.extend(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c))),
        _ => bail!("expected floats"),
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Tensors.

/// ONNX data types.
pub const FLOAT: i64 = 1;
pub const UINT8: i64 = 2;
pub const INT8: i64 = 3;
pub const INT32: i64 = 6;
pub const INT64: i64 = 7;

/// A dense tensor in f32 (integer tensors converted, their exact values kept in `ints`).
#[derive(Clone, Debug, Default)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    /// The values of an integer tensor (shapes, axes, indices, `ArgMax` results), when it is one.
    pub ints: Option<Vec<i64>>,
    /// The ONNX data type it holds (1 float, 2 uint8, 3 int8, 6 int32, 7 int64).
    pub dtype: i64,
    /// Channels-last memory (`[N, spatial..., C]` for the shape `[N, C, spatial...]`); only ever
    /// set inside [`Model::run`].
    cl: bool,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "tensor shape {shape:?} and {} values",
            data.len()
        );
        Self {
            shape,
            data,
            ints: None,
            dtype: FLOAT,
            cl: false,
        }
    }

    pub fn from_ints(shape: Vec<usize>, v: Vec<i64>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), v.len());
        Self {
            data: v.iter().map(|&x| x as f32).collect(),
            shape,
            ints: Some(v),
            dtype: INT64,
            cl: false,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The values as integers (exact for integer tensors).
    pub fn ints(&self) -> Vec<i64> {
        self.ints.clone().unwrap_or_else(|| self.data.iter().map(|&v| v as i64).collect())
    }

    fn with_layout(mut self, cl: bool) -> Self {
        self.cl = cl;
        self
    }

    /// Whether the two layouts differ in memory for this shape.
    fn layout_matters(shape: &[usize]) -> bool {
        shape.len() >= 3 && shape[1] > 1 && shape[2..].iter().product::<usize>() > 1
    }

    /// Whether the memory is channels-last and differs from `NCHW`.
    fn is_cl(&self) -> bool {
        self.cl && Self::layout_matters(&self.shape)
    }

    /// `(N, C, spatial)` of a tensor of rank 3 or more.
    fn ncs(&self) -> (usize, usize, usize) {
        (self.shape[0], self.shape[1], self.shape[2..].iter().product())
    }

    /// The same tensor in `NCHW` memory.
    pub fn into_nchw(self) -> Tensor {
        if !self.is_cl() {
            return self.with_layout(false);
        }
        let (n, c, s) = self.ncs();
        let mut out = vec![0f32; self.data.len()];
        transpose_planes(&self.data, &mut out, n, s, c);
        Tensor {
            data: out,
            cl: false,
            ..self
        }
    }

    /// The same tensor in channels-last memory.
    fn into_cl(self) -> Tensor {
        if self.cl || !Self::layout_matters(&self.shape) {
            return self.with_layout(true);
        }
        let (n, c, s) = self.ncs();
        let mut out = vec![0f32; self.data.len()];
        transpose_planes(&self.data, &mut out, n, c, s);
        Tensor {
            data: out,
            cl: true,
            ..self
        }
    }

    fn nchw_ref(&self) -> std::borrow::Cow<'_, Tensor> {
        if self.is_cl() {
            std::borrow::Cow::Owned(self.clone().into_nchw())
        } else {
            std::borrow::Cow::Borrowed(self)
        }
    }

    fn parse(b: &[u8]) -> Result<(String, Tensor)> {
        let mut p = Pb::new(b);
        let (mut dims, mut ty, mut name) = (Vec::new(), 0i64, String::new());
        let (mut fdata, mut idata, mut raw) = (Vec::new(), Vec::new(), None);
        while let Some((num, v)) = p.next()? {
            match (num, &v) {
                (1, _) => ints(&v, &mut dims)?,
                (2, Val::Varint(t)) => ty = *t as i64,
                (4, _) => floats(&v, &mut fdata)?,
                (5, _) | (7, _) => ints(&v, &mut idata)?,
                (8, Val::Bytes(s)) => name = utf8(s)?,
                (9, Val::Bytes(s)) => raw = Some(*s),
                (13, _) => bail!("tensor {name}: external data is not supported"),
                (14, Val::Varint(1)) => bail!("tensor {name}: external data is not supported"),
                _ => {}
            }
        }
        let shape: Vec<usize> = dims
            .iter()
            .map(|&d| usize::try_from(d).context("negative dimension"))
            .collect::<Result<_>>()?;
        let n: usize = shape.iter().product();
        let f32s = |r: &[u8]| r.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect::<Vec<f32>>();
        let int_vals: Option<Vec<i64>> = match (ty, raw) {
            (FLOAT, _) => None,
            (UINT8, Some(r)) => Some(r.iter().map(|&v| v as i64).collect()),
            (INT8, Some(r)) => Some(r.iter().map(|&v| v as i8 as i64).collect()),
            (INT32, Some(r)) => Some(r.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c) as i64).collect()),
            (INT64, Some(r)) => Some(r.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect()),
            // int32_data holds the INT8 / UINT8 / INT32 values, int64_data the INT64 ones.
            (UINT8 | INT8 | INT32 | INT64, None) => Some(
                idata
                    .iter()
                    .map(|&x| match ty {
                        INT8 => x as i8 as i64,
                        UINT8 => x as u8 as i64,
                        INT32 => x as i32 as i64,
                        _ => x,
                    })
                    .collect(),
            ),
            (t, _) => bail!("tensor {name}: data type {t} is not supported"),
        };
        let t = match int_vals {
            Some(v) => Tensor {
                dtype: ty,
                ..Tensor::from_ints(vec![v.len()], v)
            },
            None => Tensor::new(vec![], vec![0.0]).with_data(match raw {
                Some(r) => f32s(r),
                None => fdata,
            }),
        };
        if t.len() != n {
            bail!("tensor {name}: {} values for shape {shape:?}", t.len());
        }
        Ok((name, Tensor { shape, ..t }))
    }

    fn with_data(self, data: Vec<f32>) -> Tensor {
        Tensor { data, ..self }
    }
}

/// `src` as `[n][a][b]` to `dst` as `[n][b][a]`.
fn transpose_planes(src: &[f32], dst: &mut [f32], n: usize, a: usize, b: usize) {
    let plane = a * b;
    crate::kernels::for_chunks(dst, plane, n > 1 && plane > 4096, |i, out| {
        let s = &src[i * plane..(i + 1) * plane];
        // Blocks of 16 x 16 for the caches.
        for a0 in (0..a).step_by(16) {
            for b0 in (0..b).step_by(16) {
                for ai in a0..(a0 + 16).min(a) {
                    for bi in b0..(b0 + 16).min(b) {
                        out[bi * a + ai] = s[ai * b + bi];
                    }
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Graph.

/// An attribute value.
#[derive(Clone, Debug)]
pub enum Attr {
    Float(f32),
    Int(i64),
    Str(String),
    Tensor(Tensor),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
}

/// An element-wise step run inside the node that produces its input.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Post {
    Relu,
    /// `QuantizeLinear` then `DequantizeLinear`: `(clamp(round(x / s) + zp, lo, hi) - zp2) * s2`.
    FakeQuant {
        s: f32,
        zp: f32,
        lo: f32,
        hi: f32,
        s2: f32,
        zp2: f32,
    },
}

impl Post {
    #[inline]
    fn apply(self, v: f32) -> f32 {
        match self {
            Post::Relu => v.max(0.0),
            Post::FakeQuant { s, zp, lo, hi, s2, zp2 } => (((v / s).round_ties_even() + zp).clamp(lo, hi) - zp2) * s2,
        }
    }
}

fn apply_posts(x: &mut [f32], posts: &[Post]) {
    for p in posts {
        match *p {
            Post::Relu => x.iter_mut().for_each(|v| *v = v.max(0.0)),
            q => x.iter_mut().for_each(|v| *v = q.apply(*v)),
        }
    }
}

/// One node of the graph.
#[derive(Clone, Debug)]
pub struct Node {
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, Attr>,
    posts: Vec<Post>,
}

impl Node {
    fn parse(b: &[u8]) -> Result<Self> {
        let mut p = Pb::new(b);
        let mut n = Node {
            op: String::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            attrs: HashMap::new(),
            posts: Vec::new(),
        };
        let mut domain = String::new();
        while let Some((num, v)) = p.next()? {
            match (num, v) {
                (1, Val::Bytes(s)) => n.inputs.push(utf8(s)?),
                (2, Val::Bytes(s)) => n.outputs.push(utf8(s)?),
                (4, Val::Bytes(s)) => n.op = utf8(s)?,
                (5, Val::Bytes(s)) => {
                    let (k, a) = parse_attr(s)?;
                    n.attrs.insert(k, a);
                }
                (7, Val::Bytes(s)) => domain = utf8(s)?,
                _ => {}
            }
        }
        if !domain.is_empty() && domain != "ai.onnx" {
            bail!("operator {} of domain {domain} is not supported", n.op);
        }
        Ok(n)
    }

    fn int(&self, k: &str, default: i64) -> i64 {
        match self.attrs.get(k) {
            Some(Attr::Int(v)) => *v,
            _ => default,
        }
    }

    fn float(&self, k: &str, default: f32) -> f32 {
        match self.attrs.get(k) {
            Some(Attr::Float(v)) => *v,
            _ => default,
        }
    }

    fn ints(&self, k: &str) -> Option<Vec<i64>> {
        match self.attrs.get(k) {
            Some(Attr::Ints(v)) => Some(v.clone()),
            Some(Attr::Int(v)) => Some(vec![*v]),
            _ => None,
        }
    }

    fn str(&self, k: &str) -> Option<&str> {
        match self.attrs.get(k) {
            Some(Attr::Str(v)) => Some(v),
            _ => None,
        }
    }
}

fn parse_attr(b: &[u8]) -> Result<(String, Attr)> {
    let mut p = Pb::new(b);
    let mut name = String::new();
    let (mut f, mut i, mut s, mut t) = (None, None, None, None);
    let (mut fs, mut is) = (Vec::new(), Vec::new());
    let mut ty = 0u64;
    while let Some((num, v)) = p.next()? {
        match (num, &v) {
            (1, Val::Bytes(x)) => name = utf8(x)?,
            (2, Val::Fixed32(x)) => f = Some(f32::from_bits(*x)),
            (3, Val::Varint(x)) => i = Some(*x as i64),
            (4, Val::Bytes(x)) => s = Some(String::from_utf8_lossy(x).into_owned()),
            (5, Val::Bytes(x)) => t = Some(Tensor::parse(x)?.1),
            (7, _) => floats(&v, &mut fs)?,
            (8, _) => ints(&v, &mut is)?,
            (20, Val::Varint(x)) => ty = *x,
            _ => {}
        }
    }
    let a = match ty {
        1 => Attr::Float(f.unwrap_or(0.0)),
        2 => Attr::Int(i.unwrap_or(0)),
        3 => Attr::Str(s.unwrap_or_default()),
        4 => Attr::Tensor(t.context("tensor attribute without a tensor")?),
        6 => Attr::Floats(fs),
        7 => Attr::Ints(is),
        _ => match (f, i, s, t) {
            (Some(f), ..) => Attr::Float(f),
            (_, Some(i), ..) => Attr::Int(i),
            (_, _, Some(s), _) => Attr::Str(s),
            (.., Some(t)) => Attr::Tensor(t),
            _ if !fs.is_empty() => Attr::Floats(fs),
            _ => Attr::Ints(is),
        },
    };
    Ok((name, a))
}

/// A convolution's weights laid out for its kernel, prepared at load time.
struct ConvPlan {
    cout: usize,
    cpg: usize,
    group: usize,
    kh: usize,
    kw: usize,
    stride: [usize; 2],
    dil: [usize; 2],
    /// Top, left, bottom, right.
    pads: [usize; 4],
    one_d: bool,
    bias: Vec<f32>,
    kind: ConvKind,
}

enum ConvKind {
    /// 1 x 1, stride 1, no padding, one group: `[cout][cin]`.
    Pointwise(Vec<f32>),
    /// The same with few outputs (under 8), summed per pixel: `[cin][cout]`.
    PointwiseFew(Vec<f32>),
    /// One input and one output channel per group: `[ky][kx][c]`.
    Depthwise(Vec<f32>),
    /// One input channel (a stem on a grey image): `[ky][kx][cout]`.
    Single(Vec<f32>),
    /// im2col + GEMM: `[cout][ky][kx][ci of the group]`.
    General(Vec<f32>),
}

impl ConvPlan {
    fn new(n: &Node, w: &Tensor, b: Option<&Tensor>) -> Result<Self> {
        let one_d = w.shape.len() == 3;
        if !(one_d || w.shape.len() == 4) {
            bail!("Conv: weights of shape {:?} (1D and 2D only)", w.shape);
        }
        let (cout, cpg, kh, kw) = if one_d {
            (w.shape[0], w.shape[1], 1, w.shape[2])
        } else {
            (w.shape[0], w.shape[1], w.shape[2], w.shape[3])
        };
        let group = n.int("group", 1).max(1) as usize;
        if cout % group != 0 {
            bail!("Conv: {cout} output channels in {group} groups");
        }
        let dims = |k: &str, d: usize, len2: usize| -> Result<Vec<usize>> {
            let v = n.ints(k).unwrap_or_default();
            let v: Vec<usize> = v
                .iter()
                .map(|&x| usize::try_from(x).context("negative Conv attribute"))
                .collect::<Result<_>>()?;
            Ok(match (one_d, v.as_slice()) {
                (_, []) => vec![d; len2],
                (true, [a]) => vec![1, *a],
                (true, [a, b]) if len2 == 4 => vec![0, *a, 0, *b],
                (_, s) => s.to_vec(),
            })
        };
        let (st, dil, pads) = (dims("strides", 1, 2)?, dims("dilations", 1, 2)?, dims("pads", 0, 4)?);
        if st.len() != 2 || dil.len() != 2 || pads.len() != 4 {
            bail!("Conv: unexpected strides / dilations / pads");
        }
        match n.str("auto_pad") {
            None | Some("NOTSET") | Some("VALID") => {}
            Some(a) => bail!("Conv: auto_pad {a} is not supported"),
        }
        // ONNX pads: [x1_begin, x2_begin, x1_end, x2_end].
        let pads = [pads[0], pads[1], pads[2], pads[3]];
        let bias = match b {
            Some(b) if b.len() == cout => b.data.clone(),
            Some(b) => bail!("Conv: {} bias values for {cout} channels", b.len()),
            None => vec![0.0; cout],
        };
        let kind = if kh == 1 && kw == 1 && group == 1 && st == [1, 1] && pads == [0; 4] && cout < 8 {
            let mut wt = vec![0f32; cpg * cout];
            for c in 0..cout {
                for ci in 0..cpg {
                    wt[ci * cout + c] = w.data[c * cpg + ci];
                }
            }
            ConvKind::PointwiseFew(wt)
        } else if kh == 1 && kw == 1 && group == 1 && st == [1, 1] && pads == [0; 4] {
            ConvKind::Pointwise(w.data.clone())
        } else if cpg == 1 && group == cout {
            let mut wt = vec![0f32; kh * kw * cout];
            for c in 0..cout {
                for k in 0..kh * kw {
                    wt[k * cout + c] = w.data[c * kh * kw + k];
                }
            }
            ConvKind::Depthwise(wt)
        } else if cpg == 1 && group == 1 {
            let mut wt = vec![0f32; kh * kw * cout];
            for c in 0..cout {
                for k in 0..kh * kw {
                    wt[k * cout + c] = w.data[c * kh * kw + k];
                }
            }
            ConvKind::Single(wt)
        } else {
            let kk = kh * kw * cpg;
            let mut wp = vec![0f32; cout * kk];
            for co in 0..cout {
                for ci in 0..cpg {
                    for k in 0..kh * kw {
                        wp[co * kk + k * cpg + ci] = w.data[(co * cpg + ci) * kh * kw + k];
                    }
                }
            }
            ConvKind::General(wp)
        };
        Ok(Self {
            cout,
            cpg,
            group,
            kh,
            kw,
            stride: [st[0], st[1]],
            dil: [dil[0], dil[1]],
            pads,
            one_d,
            bias,
            kind,
        })
    }

    fn run(&self, x: Tensor, posts: &[Post]) -> Result<Tensor> {
        let x = x.into_cl();
        let (nb, cin) = (x.shape[0], x.shape[1]);
        let (h, w) = match (self.one_d, x.shape.len()) {
            (true, 3) => (1, x.shape[2]),
            (false, 4) => (x.shape[2], x.shape[3]),
            _ => bail!("Conv: input of shape {:?} for {}D weights", x.shape, if self.one_d { 1 } else { 2 }),
        };
        if cin != self.cpg * self.group {
            bail!("Conv: {cin} input channels, weights for {} x {}", self.group, self.cpg);
        }
        let [pt, pl, pb, pr] = self.pads;
        let span = |len: usize, pad: usize, k: usize, d: usize, s: usize| -> Result<usize> {
            let ext = d * (k - 1) + 1;
            if len + pad < ext {
                bail!("Conv: input of {len} too small for a kernel of {k}");
            }
            Ok((len + pad - ext) / s + 1)
        };
        let oh = span(h, pt + pb, self.kh, self.dil[0], self.stride[0])?;
        let ow = span(w, pl + pr, self.kw, self.dil[1], self.stride[1])?;
        let cout = self.cout;
        let npix = nb * oh * ow;
        let mut y = match &self.kind {
            ConvKind::Pointwise(wm) => {
                let mut y = crate::sd::ops::matmul_f32(wm, cout, cin, &x.data, npix);
                self.bias_posts(&mut y, posts);
                y
            }
            ConvKind::PointwiseFew(wt) => {
                let mut y = vec![0f32; npix * cout];
                let px = (8192 / cout).max(1);
                crate::kernels::for_chunks(&mut y, px * cout, npix * cout > 16384, |k, out| {
                    for (j, o) in out.chunks_exact_mut(cout).enumerate() {
                        let p = k * px + j;
                        o.copy_from_slice(&self.bias);
                        for (ci, &v) in x.data[p * cin..(p + 1) * cin].iter().enumerate() {
                            o.iter_mut().zip(&wt[ci * cout..(ci + 1) * cout]).for_each(|(a, &b)| *a += v * b);
                        }
                    }
                    apply_posts(out, posts);
                });
                y
            }
            ConvKind::Depthwise(wt) | ConvKind::Single(wt) => {
                let single = matches!(self.kind, ConvKind::Single(_));
                let mut y = vec![0f32; npix * cout];
                let (kh, kw) = (self.kh, self.kw);
                let ([sh, sw], [dh, dw]) = (self.stride, self.dil);
                let row = ow * cout;
                crate::kernels::for_chunks(&mut y, row, nb * oh > 1 && npix * cout > 16384, |r, out| {
                    let (i, oy) = (r / oh, r % oh);
                    // The source rows of the kernel's rows that fall inside the image.
                    let rows: Vec<(usize, &[f32])> = (0..kh)
                        .filter_map(|ky| {
                            let iy = (oy * sh + ky * dh) as isize - pt as isize;
                            (iy >= 0 && iy < h as isize).then(|| {
                                let iy = iy as usize;
                                (ky, &x.data[(i * h + iy) * w * cin..(i * h + iy + 1) * w * cin])
                            })
                        })
                        .collect();
                    // Each output pixel sums all its taps at once.
                    for (ox, o) in out.chunks_exact_mut(cout).enumerate() {
                        o.copy_from_slice(&self.bias);
                        let base = (ox * sw) as isize - pl as isize;
                        for &(ky, src) in &rows {
                            for kx in 0..kw {
                                let ix = base + (kx * dw) as isize;
                                if ix < 0 || ix >= w as isize {
                                    continue;
                                }
                                let wk = &wt[(ky * kw + kx) * cout..(ky * kw + kx + 1) * cout];
                                if single {
                                    // One input channel, broadcast to every output channel.
                                    let v = src[ix as usize];
                                    o.iter_mut().zip(wk).for_each(|(a, &b)| *a += v * b);
                                } else {
                                    let sc = &src[ix as usize * cin..ix as usize * cin + cout];
                                    o.iter_mut().zip(sc).zip(wk).for_each(|((a, &v), &b)| *a += v * b);
                                }
                            }
                        }
                    }
                    apply_posts(out, posts);
                });
                y
            }
            ConvKind::General(wp) => {
                let (cpg, kh, kw) = (self.cpg, self.kh, self.kw);
                let kk = kh * kw * cpg;
                let copg = cout / self.group;
                let ([sh, sw], [dh, dw]) = (self.stride, self.dil);
                let mut y = if self.group == 1 { Vec::new() } else { vec![0f32; npix * cout] };
                for g in 0..self.group {
                    let mut cols = vec![0f32; npix * kk];
                    crate::kernels::for_chunks(&mut cols, ow * kk, npix * kk > 65536, |r, out| {
                        let (i, oy) = (r / oh, r % oh);
                        for ky in 0..kh {
                            let iy = (oy * sh + ky * dh) as isize - pt as isize;
                            if iy < 0 || iy >= h as isize {
                                continue;
                            }
                            let src = &x.data[(i * h + iy as usize) * w * cin..(i * h + iy as usize + 1) * w * cin];
                            for kx in 0..kw {
                                let off = (kx * dw) as isize - pl as isize;
                                for ox in 0..ow {
                                    let ix = (ox * sw) as isize + off;
                                    if ix < 0 || ix >= w as isize {
                                        continue;
                                    }
                                    let s = &src[ix as usize * cin + g * cpg..ix as usize * cin + (g + 1) * cpg];
                                    out[ox * kk + (ky * kw + kx) * cpg..][..cpg].copy_from_slice(s);
                                }
                            }
                        }
                    });
                    let yg = crate::sd::ops::matmul_f32(&wp[g * copg * kk..(g + 1) * copg * kk], copg, kk, &cols, npix);
                    if self.group == 1 {
                        y = yg;
                    } else {
                        for p in 0..npix {
                            y[p * cout + g * copg..p * cout + (g + 1) * copg].copy_from_slice(&yg[p * copg..(p + 1) * copg]);
                        }
                    }
                }
                self.bias_posts(&mut y, posts);
                y
            }
        };
        y.truncate(npix * cout);
        let shape = if self.one_d { vec![nb, cout, ow] } else { vec![nb, cout, oh, ow] };
        Ok(Tensor::new(shape, y).with_layout(true))
    }

    /// Adds the bias and runs the fused steps over channels-last rows.
    fn bias_posts(&self, y: &mut [f32], posts: &[Post]) {
        let cout = self.cout;
        let rows = (16384 / cout).max(1) * cout;
        crate::kernels::for_chunks(y, rows, y.len() > 32768, |_, out| {
            for o in out.chunks_exact_mut(cout) {
                for c in 0..cout {
                    o[c] += self.bias[c];
                }
            }
            apply_posts(out, posts);
        });
    }
}

/// A loaded model: the nodes in order (constants folded, steps fused), the constants.
pub struct Model {
    nodes: Vec<Node>,
    /// Slot of each node's inputs (`None`: an omitted optional input) and outputs.
    slots_in: Vec<Vec<Option<usize>>>,
    slots_out: Vec<Vec<usize>>,
    /// Whether a node's input is used for the last time there (and is not a graph output).
    dies: Vec<Vec<bool>>,
    convs: Vec<Option<ConvPlan>>,
    /// Constants by slot.
    consts: Vec<Option<Tensor>>,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub opset: i64,
}

impl Model {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let b = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::parse(&b).with_context(|| path.display().to_string())
    }

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut p = Pb::new(bytes);
        let mut graph = None;
        let mut opset = 0;
        while let Some((num, v)) = p.next()? {
            match (num, v) {
                (7, Val::Bytes(g)) => graph = Some(g),
                (8, Val::Bytes(o)) => {
                    let mut q = Pb::new(o);
                    let (mut domain, mut ver) = (String::new(), 0);
                    while let Some((n2, v2)) = q.next()? {
                        match (n2, v2) {
                            (1, Val::Bytes(d)) => domain = utf8(d)?,
                            (2, Val::Varint(x)) => ver = x as i64,
                            _ => {}
                        }
                    }
                    if domain.is_empty() || domain == "ai.onnx" {
                        opset = ver;
                    }
                }
                _ => {}
            }
        }
        let g = graph.context("the ONNX file has no graph")?;
        let mut p = Pb::new(g);
        let (mut nodes, mut consts, mut inputs, mut outputs) = (Vec::new(), HashMap::new(), Vec::new(), Vec::new());
        let value_name = |b: &[u8]| -> Result<String> {
            let mut q = Pb::new(b);
            while let Some((n, v)) = q.next()? {
                if let (1, Val::Bytes(s)) = (n, v) {
                    return utf8(s);
                }
            }
            bail!("graph value without a name")
        };
        while let Some((num, v)) = p.next()? {
            match (num, v) {
                (1, Val::Bytes(n)) => nodes.push(Node::parse(n)?),
                (5, Val::Bytes(t)) => {
                    let (name, t) = Tensor::parse(t)?;
                    consts.insert(name, t);
                }
                (11, Val::Bytes(b)) => inputs.push(value_name(b)?),
                (12, Val::Bytes(b)) => outputs.push(value_name(b)?),
                _ => {}
            }
        }
        // Graph inputs that are initializers are constants, not inputs.
        inputs.retain(|n| !consts.contains_key(n));
        let nodes = fold(nodes, &mut consts)?;
        let nodes = fuse(nodes, &consts, &outputs);
        // Constants only the folded nodes read (int8 weights, their scales) are dropped.
        let used: std::collections::HashSet<&String> = nodes.iter().flat_map(|n| &n.inputs).chain(&outputs).collect();
        consts.retain(|k, _| used.contains(k));
        Self::compile(nodes, consts, inputs, outputs, opset)
    }

    fn compile(nodes: Vec<Node>, consts: HashMap<String, Tensor>, inputs: Vec<String>, outputs: Vec<String>, opset: i64) -> Result<Self> {
        let mut slot: HashMap<String, usize> = HashMap::new();
        let mut const_slots: Vec<Option<Tensor>> = Vec::new();
        let id = |name: &str, slot: &mut HashMap<String, usize>, cs: &mut Vec<Option<Tensor>>| -> usize {
            if let Some(&s) = slot.get(name) {
                return s;
            }
            let s = cs.len();
            slot.insert(name.to_string(), s);
            cs.push(None);
            s
        };
        for (k, t) in consts {
            let s = id(&k, &mut slot, &mut const_slots);
            const_slots[s] = Some(t);
        }
        let input_slots: Vec<usize> = inputs.iter().map(|n| id(n, &mut slot, &mut const_slots)).collect();
        let mut slots_in = Vec::new();
        let mut slots_out = Vec::new();
        let mut convs = Vec::new();
        for n in &nodes {
            let ins: Vec<Option<usize>> = n
                .inputs
                .iter()
                .map(|x| (!x.is_empty()).then(|| id(x, &mut slot, &mut const_slots)))
                .collect();
            let outs: Vec<usize> = n.outputs.iter().map(|x| id(x, &mut slot, &mut const_slots)).collect();
            convs.push(if n.op == "Conv" {
                let w = ins.get(1).copied().flatten().and_then(|s| const_slots[s].as_ref());
                let b = ins.get(2).copied().flatten().and_then(|s| const_slots[s].as_ref());
                let w = w.context("Conv with computed weights is not supported")?;
                Some(ConvPlan::new(n, w, b)?)
            } else {
                None
            });
            slots_in.push(ins);
            slots_out.push(outs);
        }
        let output_slots: Vec<usize> = outputs.iter().map(|n| id(n, &mut slot, &mut const_slots)).collect();
        // Every value is computed before use.
        let mut known: Vec<bool> = const_slots.iter().map(Option::is_some).collect();
        for &s in &input_slots {
            known[s] = true;
        }
        for (i, n) in nodes.iter().enumerate() {
            for (k, s) in slots_in[i].iter().enumerate() {
                if let Some(s) = s
                    && !known[*s]
                {
                    bail!("{}: input {} is not computed before it", n.op, n.inputs[k]);
                }
            }
            for &s in &slots_out[i] {
                known[s] = true;
            }
        }
        if let Some(o) = output_slots.iter().position(|&s| !known[s]) {
            bail!("graph output {} is never computed", outputs[o]);
        }
        let mut last = vec![usize::MAX; const_slots.len()];
        for (i, ins) in slots_in.iter().enumerate() {
            for s in ins.iter().flatten() {
                last[*s] = i;
            }
        }
        let dies = slots_in
            .iter()
            .enumerate()
            .map(|(i, ins)| {
                ins.iter()
                    .map(|s| {
                        s.is_some_and(|s| {
                            last[s] == i
                                && const_slots[s].is_none()
                                && !output_slots.contains(&s)
                                && ins.iter().filter(|x| **x == Some(s)).count() == 1
                        })
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            nodes,
            slots_in,
            slots_out,
            dies,
            convs,
            consts: const_slots,
            input_slots,
            output_slots,
            inputs,
            outputs,
            opset,
        })
    }

    /// The operator names of the graph after folding and fusion, in order.
    /// The number of constant values (weights) the graph keeps after folding.
    pub fn params(&self) -> usize {
        self.consts.iter().flatten().map(Tensor::len).sum()
    }

    pub fn ops(&self) -> Vec<&str> {
        self.nodes.iter().map(|n| n.op.as_str()).collect()
    }

    /// The index of output `name`.
    pub fn output_index(&self, name: &str) -> Option<usize> {
        self.outputs.iter().position(|o| o == name)
    }

    /// Runs the graph on `inputs` (by name); returns the graph's outputs in order (`NCHW`).
    pub fn run(&self, inputs: Vec<(&str, Tensor)>) -> Result<Vec<Tensor>> {
        let mut vals: Vec<Option<Tensor>> = vec![None; self.consts.len()];
        for (k, t) in inputs {
            let i = self
                .inputs
                .iter()
                .position(|n| n == k)
                .with_context(|| format!("the model has no input {k}"))?;
            vals[self.input_slots[i]] = Some(t.with_layout(false));
        }
        for (i, &s) in self.input_slots.iter().enumerate() {
            if vals[s].is_none() {
                bail!("input {} is missing", self.inputs[i]);
            }
        }
        for (i, n) in self.nodes.iter().enumerate() {
            // Inputs used for the last time here are handed over (run in place where possible).
            let mut owned: Vec<Option<Tensor>> = self.slots_in[i]
                .iter()
                .zip(&self.dies[i])
                .map(|(s, &d)| match s {
                    Some(s) if d => vals[*s].take(),
                    _ => None,
                })
                .collect();
            let refs: Vec<Option<&Tensor>> = self.slots_in[i]
                .iter()
                .zip(&owned)
                .map(|(s, o)| match (s, o) {
                    (_, Some(_)) => None,
                    (Some(s), None) => vals[*s].as_ref().or(self.consts[*s].as_ref()),
                    (None, None) => None,
                })
                .collect();
            let mut ins: Vec<Arg> = Vec::with_capacity(refs.len());
            for (k, r) in refs.into_iter().enumerate() {
                ins.push(match (owned[k].take(), r) {
                    (Some(t), _) => Arg::Own(t),
                    (None, Some(t)) => Arg::Ref(t),
                    (None, None) => Arg::None,
                });
            }
            let t0 = profile().then(std::time::Instant::now);
            let outs = run_node(n, ins, self.convs[i].as_ref()).with_context(|| format!("{} -> {:?}", n.op, n.outputs))?;
            if let Some(t0) = t0 {
                let kind = match &self.convs[i] {
                    Some(c) => {
                        let o = &outs[0];
                        let k = match c.kind {
                            ConvKind::Pointwise(_) | ConvKind::PointwiseFew(_) => "pointwise",
                            ConvKind::Depthwise(_) => "depthwise",
                            ConvKind::Single(_) => "one-channel",
                            ConvKind::General(_) => "im2col",
                        };
                        format!(
                            "Conv {k} {}x{} {}->{} s{} @{:?}",
                            c.kh,
                            c.kw,
                            c.cpg * c.group,
                            c.cout,
                            c.stride[1],
                            &o.shape[2..]
                        )
                    }
                    None => n.op.clone(),
                };
                let mut p = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
                let e = p.entry(kind).or_default();
                e.0 += 1;
                e.1 += t0.elapsed().as_secs_f64();
            }
            if outs.len() < self.slots_out[i].len() {
                bail!("{}: {} outputs computed, {} expected", n.op, outs.len(), self.slots_out[i].len());
            }
            for (&s, t) in self.slots_out[i].iter().zip(outs) {
                vals[s] = Some(t);
            }
        }
        self.output_slots
            .iter()
            .zip(&self.outputs)
            .map(|(&s, name)| {
                let t = match vals[s].take() {
                    Some(t) => t,
                    None => self.consts[s].clone().with_context(|| format!("output {name} not computed"))?,
                };
                Ok(t.into_nchw())
            })
            .collect()
    }
}

/// Time per operator kind while [`set_profile`] is on: calls and seconds, printed by
/// [`profile_report`].
static PROFILE: std::sync::Mutex<std::collections::BTreeMap<String, (usize, f64)>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());
static PROFILE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn profile() -> bool {
    PROFILE_ON.load(std::sync::atomic::Ordering::Relaxed)
}

/// Times every operator run from now on (process-wide), for [`profile_report`].
pub fn set_profile(on: bool) {
    PROFILE_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The time per operator kind since the last report (while [`set_profile`] is on), as lines.
pub fn profile_report() -> String {
    let mut p = PROFILE.lock().unwrap_or_else(|e| e.into_inner());
    let mut v: Vec<(String, (usize, f64))> = std::mem::take(&mut *p).into_iter().collect();
    v.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    v.iter()
        .map(|(k, (n, t))| format!("{k:>44}: {n:6} calls {:9.2} ms\n", t * 1e3))
        .collect()
}

/// Runs every node whose inputs are all constants once, now (`Constant` nodes, weights through
/// `DequantizeLinear`, shape constants), keeping the others.
fn fold(nodes: Vec<Node>, consts: &mut HashMap<String, Tensor>) -> Result<Vec<Node>> {
    let mut keep = Vec::new();
    for n in nodes {
        let constant = n.inputs.iter().all(|i| i.is_empty() || consts.contains_key(i));
        if constant {
            let ins: Vec<Arg> = n
                .inputs
                .iter()
                .map(|i| match consts.get(i) {
                    Some(t) => Arg::Ref(t),
                    None => Arg::None,
                })
                .collect();
            let outs = run_node(&n, ins, None).with_context(|| format!("fold {} -> {:?}", n.op, n.outputs))?;
            for (name, t) in n.outputs.iter().zip(outs) {
                consts.insert(name.clone(), t.into_nchw());
            }
        } else {
            keep.push(n);
        }
    }
    Ok(keep)
}

/// A per-tensor quantization's (scale, zero point, range) from a node's constant inputs.
fn qparams_const(n: &Node, consts: &HashMap<String, Tensor>) -> Option<(f32, f32, f32, f32)> {
    let s = consts.get(n.inputs.get(1)?)?;
    if s.len() != 1 {
        return None;
    }
    let z = n.inputs.get(2).filter(|z| !z.is_empty()).map(|z| consts.get(z));
    let (zp, signed) = match z {
        None => (0.0, false),
        Some(Some(z)) if z.len() == 1 => (z.data[0], z.dtype == INT8),
        Some(_) => return None,
    };
    let (lo, hi) = if signed { (-128.0, 127.0) } else { (0.0, 255.0) };
    Some((s.data[0], zp, lo, hi))
}

/// `QuantizeLinear` + `DequantizeLinear` pairs into one step, then `Relu` and fake quantization
/// into the `Conv` / `Add` producing their input when nothing else reads it.
fn fuse(nodes: Vec<Node>, consts: &HashMap<String, Tensor>, outputs: &[String]) -> Vec<Node> {
    let uses = |nodes: &[Node]| {
        let mut u: HashMap<String, usize> = HashMap::new();
        for n in nodes {
            for i in &n.inputs {
                *u.entry(i.clone()).or_default() += 1;
            }
        }
        for o in outputs {
            *u.entry(o.clone()).or_default() += 1;
        }
        u
    };
    // Q -> DQ.
    let u = uses(&nodes);
    let mut out: Vec<Node> = Vec::with_capacity(nodes.len());
    let mut skip = vec![false; nodes.len()];
    for i in 0..nodes.len() {
        if skip[i] {
            continue;
        }
        let n = &nodes[i];
        if n.op == "QuantizeLinear"
            && u.get(&n.outputs[0]) == Some(&1)
            && let Some((s, zp, lo, hi)) = qparams_const(n, consts)
            && let Some(j) = (i + 1..nodes.len()).find(|&j| nodes[j].inputs.first() == Some(&n.outputs[0]))
            && nodes[j].op == "DequantizeLinear"
            && let Some((s2, zp2, ..)) = qparams_const(&nodes[j], consts)
        {
            skip[j] = true;
            out.push(Node {
                op: "FakeQuant".into(),
                inputs: vec![n.inputs[0].clone()],
                outputs: nodes[j].outputs.clone(),
                attrs: HashMap::new(),
                posts: vec![Post::FakeQuant { s, zp, lo, hi, s2, zp2 }],
            });
            continue;
        }
        out.push(n.clone());
    }
    // Element-wise steps into their producer.
    let u = uses(&out);
    let mut producer: HashMap<String, usize> = HashMap::new();
    let mut removed = vec![false; out.len()];
    for i in 0..out.len() {
        let post = match out[i].op.as_str() {
            "Relu" => Some(Post::Relu),
            "FakeQuant" => Some(out[i].posts[0]),
            _ => None,
        };
        if let Some(post) = post
            && let Some(&p) = producer.get(&out[i].inputs[0])
            && matches!(out[p].op.as_str(), "Conv" | "Add" | "FakeQuant")
            && u.get(&out[i].inputs[0]) == Some(&1)
        {
            let name = out[i].outputs[0].clone();
            out[p].posts.push(post);
            out[p].outputs[0] = name.clone();
            producer.insert(name, p);
            removed[i] = true;
            continue;
        }
        for o in &out[i].outputs {
            producer.insert(o.clone(), i);
        }
    }
    out.into_iter().zip(removed).filter(|(_, r)| !r).map(|(n, _)| n).collect()
}

// ---------------------------------------------------------------------------------------------
// Operators.

/// A node input: handed over (last use), borrowed, or omitted.
enum Arg<'a> {
    Own(Tensor),
    Ref(&'a Tensor),
    None,
}

impl Arg<'_> {
    fn get(&self) -> Option<&Tensor> {
        match self {
            Arg::Own(t) => Some(t),
            Arg::Ref(t) => Some(t),
            Arg::None => None,
        }
    }

    fn take(self) -> Option<Tensor> {
        match self {
            Arg::Own(t) => Some(t),
            Arg::Ref(t) => Some(t.clone()),
            Arg::None => None,
        }
    }
}

fn input<'a>(ins: &'a [Arg], i: usize, op: &str) -> Result<&'a Tensor> {
    ins.get(i).and_then(Arg::get).with_context(|| format!("{op}: input {i} missing"))
}

fn take_input(ins: &mut [Arg], i: usize, op: &str) -> Result<Tensor> {
    let a = std::mem::replace(ins.get_mut(i).with_context(|| format!("{op}: input {i} missing"))?, Arg::None);
    a.take().with_context(|| format!("{op}: input {i} missing"))
}

/// The shape both operands broadcast to (numpy rules).
fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
            let x = if i + a.len() >= n { a[i + a.len() - n] } else { 1 };
            let y = if i + b.len() >= n { b[i + b.len() - n] } else { 1 };
            match (x, y) {
                _ if x == y => Ok(x),
                (1, _) => Ok(y),
                (_, 1) => Ok(x),
                _ => bail!("shapes {a:?} and {b:?} do not broadcast"),
            }
        })
        .collect()
}

fn binary(ins: &mut [Arg], op: &str, posts: &[Post], f: impl Fn(f32, f32) -> f32 + Sync + Copy) -> Result<Tensor> {
    let (sa, sb) = (input(ins, 0, op)?.shape.clone(), input(ins, 1, op)?.shape.clone());
    if sa == sb {
        // Same shape: element by element in either layout (both brought to the first one's).
        let cl = input(ins, 0, op)?.is_cl() || input(ins, 1, op)?.is_cl();
        let fix = |t: Tensor| if cl { t.into_cl() } else { t.into_nchw() };
        let (mut a, b, swapped) = match (std::mem::replace(&mut ins[0], Arg::None), std::mem::replace(&mut ins[1], Arg::None)) {
            (Arg::Own(a), b) => (fix(a), b.take().context("binary input")?, false),
            (a, Arg::Own(b)) => (fix(b), a.take().context("binary input")?, true),
            (a, b) => (fix(a.take().context("binary input")?), b.take().context("binary input")?, false),
        };
        let b = fix(b);
        let par = a.len() > 65536;
        crate::kernels::for_chunks(&mut a.data, 16384, par, |k, c| {
            let bs = &b.data[k * 16384..k * 16384 + c.len()];
            if swapped {
                c.iter_mut().zip(bs).for_each(|(x, &y)| *x = f(y, *x));
            } else {
                c.iter_mut().zip(bs).for_each(|(x, &y)| *x = f(*x, y));
            }
            apply_posts(c, posts);
        });
        a.ints = None;
        a.dtype = FLOAT;
        return Ok(a);
    }
    let (a, b) = (input(ins, 0, op)?.nchw_ref(), input(ins, 1, op)?.nchw_ref());
    let shape = broadcast(&a.shape, &b.shape)?;
    let rank = shape.len();
    let strides = |s: &[usize]| -> Vec<usize> {
        let pad = rank - s.len();
        let mut st = vec![0usize; rank];
        let mut acc = 1;
        for i in (0..s.len()).rev() {
            st[pad + i] = if s[i] == 1 { 0 } else { acc };
            acc *= s[i];
        }
        st
    };
    let (sta, stb) = (strides(&a.shape), strides(&b.shape));
    let inner = *shape.last().unwrap_or(&1);
    let n: usize = shape.iter().product();
    let mut data = vec![0f32; n];
    if n > 0 {
        let (ia, ib) = (sta.last().copied().unwrap_or(0), stb.last().copied().unwrap_or(0));
        crate::kernels::for_chunks(&mut data, inner.max(1), n > 65536, |row, out| {
            // Offsets of this row (all dims but the last).
            let (mut oa, mut ob, mut r) = (0usize, 0usize, row);
            for d in (0..rank.saturating_sub(1)).rev() {
                let i = r % shape[d];
                r /= shape[d];
                oa += i * sta[d];
                ob += i * stb[d];
            }
            let n = out.len();
            match (ia, ib) {
                (1, 1) => out
                    .iter_mut()
                    .zip(&a.data[oa..oa + n])
                    .zip(&b.data[ob..ob + n])
                    .for_each(|((o, &x), &y)| *o = f(x, y)),
                (1, 0) => {
                    let y = b.data[ob];
                    out.iter_mut().zip(&a.data[oa..oa + n]).for_each(|(o, &x)| *o = f(x, y));
                }
                (0, 1) => {
                    let x = a.data[oa];
                    out.iter_mut().zip(&b.data[ob..ob + n]).for_each(|(o, &y)| *o = f(x, y));
                }
                _ => {
                    for (x, o) in out.iter_mut().enumerate() {
                        *o = f(a.data[oa + x * ia], b.data[ob + x * ib]);
                    }
                }
            }
            apply_posts(out, posts);
        });
    }
    Ok(Tensor::new(shape, data))
}

fn unary(ins: &mut [Arg], op: &str, f: impl Fn(f32) -> f32 + Sync) -> Result<Tensor> {
    let mut a = take_input(ins, 0, op)?;
    let par = a.len() > 65536;
    crate::kernels::for_chunks(&mut a.data, 16384, par, |_, c| c.iter_mut().for_each(|v| *v = f(*v)));
    a.ints = None;
    a.dtype = FLOAT;
    Ok(a)
}

fn norm_axis(axis: i64, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    if a < 0 || a as usize >= rank.max(1) {
        bail!("axis {axis} out of range for rank {rank}");
    }
    Ok(a as usize)
}

#[derive(Clone, Copy)]
enum Red {
    Sum,
    Mean,
    Max,
    LogSumExp,
}

/// A reduction of `x` viewed as `[outer][len][inner]` over `len`.
fn reduce_raw(x: &[f32], outer: usize, len: usize, inner: usize, kind: Red) -> Vec<f32> {
    let mut out = vec![0f32; outer * inner];
    let par = outer * inner * len > 65536;
    if inner == 1 {
        let chunk = (4096 / len.max(1)).max(1);
        crate::kernels::for_chunks(&mut out, chunk, par, |k, o| {
            for (j, v) in o.iter_mut().enumerate() {
                let r = &x[(k * chunk + j) * len..(k * chunk + j + 1) * len];
                *v = match kind {
                    Red::Sum => r.iter().sum(),
                    Red::Mean => r.iter().sum::<f32>() / len as f32,
                    Red::Max => r.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                    Red::LogSumExp => {
                        let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        if m == f32::NEG_INFINITY {
                            m
                        } else {
                            m + r.iter().map(|v| (v - m).exp()).sum::<f32>().ln()
                        }
                    }
                };
            }
        });
        return out;
    }
    // Blocks of each outer row's inner values, the reduced axis outermost (contiguous reads).
    const B: usize = 4096;
    for (oi, row) in out.chunks_exact_mut(inner).enumerate() {
        let base = &x[oi * len * inner..(oi + 1) * len * inner];
        crate::kernels::for_chunks(row, B, par && inner > B, |k, o| {
            let i0 = k * B;
            let n = o.len();
            let at = |l: usize| &base[l * inner + i0..l * inner + i0 + n];
            match kind {
                Red::Sum | Red::Mean => {
                    for l in 0..len {
                        o.iter_mut().zip(at(l)).for_each(|(a, b)| *a += b);
                    }
                    if let Red::Mean = kind {
                        o.iter_mut().for_each(|a| *a /= len as f32);
                    }
                }
                Red::Max | Red::LogSumExp => {
                    o.fill(f32::NEG_INFINITY);
                    for l in 0..len {
                        o.iter_mut().zip(at(l)).for_each(|(a, b)| *a = a.max(*b));
                    }
                    if let Red::LogSumExp = kind {
                        let mut s = vec![0f32; n];
                        for l in 0..len {
                            s.iter_mut().zip(at(l)).zip(o.iter()).for_each(|((s, v), m)| *s += (v - m).exp());
                        }
                        for (m, s) in o.iter_mut().zip(&s) {
                            if *m != f32::NEG_INFINITY {
                                *m += s.ln();
                            }
                        }
                    }
                }
            }
        });
    }
    out
}

/// `ArgMax` of `x` viewed as `[outer][len][inner]` over `len` (the first maximum unless `last`).
fn argmax_raw(x: &[f32], outer: usize, len: usize, inner: usize, last: bool) -> Vec<i64> {
    let mut out = vec![0i64; outer * inner];
    let better = |c: f32, b: f32| c > b || (last && c == b);
    if inner == 1 {
        // Indices as f32 (exact below 2^24) through the parallel chunks.
        let chunk = (4096 / len.max(1)).max(1);
        let mut idx = vec![0f32; outer];
        crate::kernels::for_chunks(&mut idx, chunk, outer * len > 65536, |k, o| {
            for (j, v) in o.iter_mut().enumerate() {
                let r = &x[(k * chunk + j) * len..(k * chunk + j + 1) * len];
                let mut best = 0usize;
                for (i, &c) in r.iter().enumerate().skip(1) {
                    if better(c, r[best]) {
                        best = i;
                    }
                }
                *v = best as f32;
            }
        });
        out.iter_mut().zip(&idx).for_each(|(o, &v)| *o = v as i64);
        return out;
    }
    for o in 0..outer {
        for i in 0..inner {
            let mut best = 0usize;
            for k in 1..len {
                if better(x[(o * len + k) * inner + i], x[(o * len + best) * inner + i]) {
                    best = k;
                }
            }
            out[o * inner + i] = best as i64;
        }
    }
    out
}

/// The reduction axes of a `Reduce*` node (attribute before opset 18, else the second input),
/// sorted, or every axis.
fn reduce_axes(n: &Node, ins: &[Arg], rank: usize) -> Result<Vec<usize>> {
    let axes = n.ints("axes").or_else(|| ins.get(1).and_then(Arg::get).map(Tensor::ints));
    let mut v: Vec<usize> = match axes {
        Some(a) if !a.is_empty() => a.iter().map(|&x| norm_axis(x, rank)).collect::<Result<_>>()?,
        _ if n.int("noop_with_empty_axes", 0) != 0 => Vec::new(),
        _ => (0..rank).collect(),
    };
    v.sort_unstable();
    v.dedup();
    Ok(v)
}

fn run_node(n: &Node, mut ins: Vec<Arg>, conv: Option<&ConvPlan>) -> Result<Vec<Tensor>> {
    let op = n.op.as_str();
    Ok(vec![match op {
        "Constant" => match n.attrs.get("value") {
            Some(Attr::Tensor(t)) => t.clone(),
            _ => match (
                n.attrs.get("value_float"),
                n.attrs.get("value_ints"),
                n.attrs.get("value_int"),
                n.attrs.get("value_floats"),
            ) {
                (Some(Attr::Float(f)), ..) => Tensor::new(vec![], vec![*f]),
                (_, Some(Attr::Ints(v)), ..) => Tensor::from_ints(vec![v.len()], v.clone()),
                (_, _, Some(Attr::Int(v)), _) => Tensor::from_ints(vec![], vec![*v]),
                (.., Some(Attr::Floats(v))) => Tensor::new(vec![v.len()], v.clone()),
                _ => bail!("Constant: unsupported value"),
            },
        },
        "Identity" => take_input(&mut ins, 0, op)?,
        "Relu" => unary(&mut ins, op, |v| v.max(0.0))?,
        "Sigmoid" => unary(&mut ins, op, |v| 1.0 / (1.0 + (-v).exp()))?,
        "Exp" => unary(&mut ins, op, f32::exp)?,
        "Tanh" => unary(&mut ins, op, f32::tanh)?,
        "FakeQuant" => {
            let mut a = take_input(&mut ins, 0, op)?;
            let posts = n.posts.clone();
            let par = a.data.len() > 65536;
            crate::kernels::for_chunks(&mut a.data, 16384, par, |_, c| apply_posts(c, &posts));
            a.ints = None;
            a.dtype = FLOAT;
            return Ok(vec![a]);
        }
        "Add" => return Ok(vec![binary(&mut ins, op, &n.posts, |a, b| a + b)?]),
        "Sub" => binary(&mut ins, op, &[], |a, b| a - b)?,
        "Mul" => binary(&mut ins, op, &[], |a, b| a * b)?,
        "Div" => binary(&mut ins, op, &[], |a, b| a / b)?,
        "ReduceSum" | "ReduceMean" | "ReduceMax" | "ReduceLogSumExp" => {
            let kind = match op {
                "ReduceSum" => Red::Sum,
                "ReduceMean" => Red::Mean,
                "ReduceMax" => Red::Max,
                _ => Red::LogSumExp,
            };
            let a = input(&ins, 0, op)?;
            let keep = n.int("keepdims", 1) != 0;
            let axes = reduce_axes(n, &ins, a.shape.len())?;
            if axes.is_empty() {
                return Ok(vec![take_input(&mut ins, 0, op)?]);
            }
            let mut shape = a.shape.clone();
            for &ax in &axes {
                shape[ax] = 1;
            }
            let data = if a.is_cl() && axes == [1] {
                // Over the contiguous channels.
                let (nb, c, s) = a.ncs();
                reduce_raw(&a.data, nb * s, c, 1, kind)
            } else if axes.windows(2).all(|w| w[1] == w[0] + 1) {
                // Contiguous axes: one reduction.
                let a = a.nchw_ref();
                let outer: usize = a.shape[..axes[0]].iter().product();
                let len: usize = axes.iter().map(|&x| a.shape[x]).product();
                let inner: usize = a.shape[axes[axes.len() - 1] + 1..].iter().product();
                reduce_raw(&a.data, outer, len, inner, kind)
            } else {
                // One axis at a time (LogSumExp over separate axes is not that).
                if let Red::LogSumExp = kind {
                    bail!("ReduceLogSumExp over non-adjacent axes is not supported");
                }
                let a = a.nchw_ref();
                let (mut cur, mut sh) = (a.data.clone(), a.shape.clone());
                for &ax in axes.iter().rev() {
                    let outer: usize = sh[..ax].iter().product();
                    let inner: usize = sh[ax + 1..].iter().product();
                    let k = if let Red::Mean = kind { Red::Sum } else { kind };
                    cur = reduce_raw(&cur, outer, sh[ax], inner, k);
                    sh[ax] = 1;
                }
                if let Red::Mean = kind {
                    let cnt: usize = axes.iter().map(|&x| a.shape[x]).product();
                    cur.iter_mut().for_each(|v| *v /= cnt as f32);
                }
                cur
            };
            if !keep {
                shape = shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(i))
                    .map(|(_, &d)| d)
                    .collect();
            }
            Tensor::new(shape, data)
        }
        "ArgMax" => {
            let a = input(&ins, 0, op)?;
            let ax = norm_axis(n.int("axis", 0), a.shape.len())?;
            let keep = n.int("keepdims", 1) != 0;
            let last = n.int("select_last_index", 0) != 0;
            let v = if a.is_cl() && ax == 1 {
                let (nb, c, s) = a.ncs();
                argmax_raw(&a.data, nb * s, c, 1, last)
            } else {
                let a = a.nchw_ref();
                let outer: usize = a.shape[..ax].iter().product();
                let inner: usize = a.shape[ax + 1..].iter().product();
                argmax_raw(&a.data, outer, a.shape[ax], inner, last)
            };
            let mut shape = a.shape.clone();
            if keep {
                shape[ax] = 1;
            } else {
                shape.remove(ax);
            }
            Tensor::from_ints(shape, v)
        }
        "Squeeze" | "Unsqueeze" => {
            let axes = n.ints("axes").or_else(|| ins.get(1).and_then(Arg::get).map(Tensor::ints));
            let a = take_input(&mut ins, 0, op)?;
            let rank = a.shape.len();
            let mut shape = a.shape.clone();
            // Channels-last stays so when only spatial axes come and go.
            let keep_cl;
            if op == "Squeeze" {
                let ax: Vec<usize> = match axes {
                    Some(ax) => {
                        let mut ax: Vec<usize> = ax.iter().map(|&v| norm_axis(v, rank)).collect::<Result<_>>()?;
                        ax.sort_unstable();
                        ax.dedup();
                        ax
                    }
                    None => (0..rank).filter(|&i| shape[i] == 1).collect(),
                };
                if let Some(&bad) = ax.iter().find(|&&i| shape[i] != 1) {
                    bail!("Squeeze: axis {bad} of {:?} is not 1", a.shape);
                }
                keep_cl = ax.iter().all(|&i| i >= 2) && rank - ax.len() >= 3;
                for i in ax.iter().rev() {
                    shape.remove(*i);
                }
            } else {
                let ax = axes.context("Unsqueeze without axes")?;
                let out_rank = rank + ax.len();
                let mut ax: Vec<usize> = ax.iter().map(|&v| norm_axis(v, out_rank)).collect::<Result<_>>()?;
                ax.sort_unstable();
                keep_cl = ax.iter().all(|&i| i >= 2) && rank >= 3;
                for a in ax {
                    shape.insert(a, 1);
                }
            }
            let a = if keep_cl { a } else { a.into_nchw() };
            let cl = a.cl;
            Tensor { shape, ..a }.with_layout(cl)
        }
        "Concat" => {
            let parts: Vec<&Tensor> = (0..ins.len()).map(|i| input(&ins, i, op)).collect::<Result<_>>()?;
            let rank = parts[0].shape.len();
            let ax = norm_axis(n.int("axis", 0), rank)?;
            for p in &parts {
                if p.shape.len() != rank || p.shape.iter().enumerate().any(|(i, &d)| i != ax && d != parts[0].shape[i]) {
                    bail!("Concat: shapes {:?} and {:?}", parts[0].shape, p.shape);
                }
            }
            let mut shape = parts[0].shape.clone();
            shape[ax] = parts.iter().map(|p| p.shape[ax]).sum();
            let int = parts.iter().all(|p| p.ints.is_some());
            if ax == 1 && rank >= 3 && parts.iter().any(|p| p.is_cl()) {
                // Channels-last: each pixel's channels side by side.
                let cl: Vec<Tensor> = parts.iter().map(|p| (*p).clone().into_cl()).collect();
                let (nb, _, s) = cl[0].ncs();
                let ctot = shape[1];
                let mut data = vec![0f32; nb * s * ctot];
                crate::kernels::for_chunks(&mut data, ctot * 256, nb * s * ctot > 65536, |k, out| {
                    for (j, o) in out.chunks_exact_mut(ctot).enumerate() {
                        let px = k * 256 + j;
                        let mut c0 = 0;
                        for p in &cl {
                            let c = p.shape[1];
                            o[c0..c0 + c].copy_from_slice(&p.data[px * c..(px + 1) * c]);
                            c0 += c;
                        }
                    }
                });
                return Ok(vec![Tensor::new(shape, data).with_layout(true)]);
            }
            let parts: Vec<std::borrow::Cow<Tensor>> = parts.iter().map(|p| p.nchw_ref()).collect();
            let outer: usize = parts[0].shape[..ax].iter().product();
            let mut data = Vec::with_capacity(shape.iter().product());
            let mut iv = Vec::new();
            for o in 0..outer {
                for p in &parts {
                    let blk: usize = p.shape[ax..].iter().product();
                    data.extend_from_slice(&p.data[o * blk..(o + 1) * blk]);
                    if int {
                        iv.extend_from_slice(&p.ints.as_ref().unwrap()[o * blk..(o + 1) * blk]);
                    }
                }
            }
            if int {
                Tensor::from_ints(shape, iv)
            } else {
                Tensor::new(shape, data)
            }
        }
        "Gemm" => {
            let (a, b) = (input(&ins, 0, op)?, input(&ins, 1, op)?);
            if a.shape.len() != 2 || b.shape.len() != 2 {
                bail!("Gemm: shapes {:?} and {:?}", a.shape, b.shape);
            }
            let (ta, tb) = (n.int("transA", 0) != 0, n.int("transB", 0) != 0);
            let (alpha, beta) = (n.float("alpha", 1.0), n.float("beta", 1.0));
            let (m, k) = if ta { (a.shape[1], a.shape[0]) } else { (a.shape[0], a.shape[1]) };
            let (kb, nn) = if tb { (b.shape[1], b.shape[0]) } else { (b.shape[0], b.shape[1]) };
            if k != kb {
                bail!("Gemm: inner sizes {k} and {kb}");
            }
            let c = ins.get(2).and_then(Arg::get);
            let mut data = vec![0f32; m * nn];
            for i in 0..m {
                for j in 0..nn {
                    let mut s = 0f32;
                    for l in 0..k {
                        let av = if ta { a.data[l * m + i] } else { a.data[i * k + l] };
                        let bv = if tb { b.data[j * k + l] } else { b.data[l * nn + j] };
                        s += av * bv;
                    }
                    let cv = c.map_or(0.0, |c| match c.len() {
                        1 => c.data[0],
                        l if l == nn => c.data[j],
                        l if l == m => c.data[i],
                        _ => c.data[i * nn + j],
                    });
                    data[i * nn + j] = alpha * s + beta * cv;
                }
            }
            Tensor::new(vec![m, nn], data)
        }
        "Resize" => resize_nearest(n, &ins)?,
        "Conv" => {
            let plan = conv.context("Conv without a plan")?;
            let x = take_input(&mut ins, 0, op)?;
            return Ok(vec![plan.run(x, &n.posts)?]);
        }
        "QuantizeLinear" | "DequantizeLinear" => {
            let a = input(&ins, 0, op)?;
            let s = input(&ins, 1, op)?;
            let z = ins.get(2).and_then(Arg::get);
            let axis = n.int("axis", 1);
            let per_axis = s.len() > 1;
            let a = if per_axis { a.nchw_ref() } else { std::borrow::Cow::Borrowed(a) };
            let (len, inner) = if per_axis {
                let ax = norm_axis(axis, a.shape.len())?;
                if a.shape[ax] != s.len() {
                    bail!("{op}: {} scales for axis {ax} of {:?}", s.len(), a.shape);
                }
                (a.shape[ax], a.shape[ax + 1..].iter().product::<usize>())
            } else {
                (1, 1)
            };
            let param = |i: usize| -> (f32, f32) {
                let c = if per_axis { (i / inner) % len } else { 0 };
                (s.data[c], z.map_or(0.0, |z| z.data[if z.len() > 1 { c } else { 0 }]))
            };
            let cl = a.cl;
            if op == "QuantizeLinear" {
                // The zero point's type sets the range: uint8 (also without one) or int8.
                let signed = z.is_some_and(|z| z.dtype == INT8);
                let (lo, hi) = if signed { (-128.0, 127.0) } else { (0.0, 255.0) };
                let v: Vec<i64> = a
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let (sc, zp) = param(i);
                        ((x / sc).round_ties_even() + zp).clamp(lo, hi) as i64
                    })
                    .collect();
                let mut t = Tensor::from_ints(a.shape.clone(), v).with_layout(cl);
                t.dtype = if signed { INT8 } else { UINT8 };
                t
            } else {
                let data: Vec<f32> = a
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        let (sc, zp) = param(i);
                        (x - zp) * sc
                    })
                    .collect();
                Tensor::new(a.shape.clone(), data).with_layout(cl)
            }
        }
        _ => bail!("ONNX operator {op} is not supported"),
    }])
    .map(|mut v: Vec<Tensor>| {
        if !n.posts.is_empty() && !matches!(op, "Conv" | "Add" | "FakeQuant") {
            for t in &mut v {
                apply_posts(&mut t.data, &n.posts);
            }
        }
        v
    })
}

/// `Resize` in mode `nearest` (opset 11+ inputs: X, roi, scales, sizes) over the last two axes.
fn resize_nearest(n: &Node, ins: &[Arg]) -> Result<Tensor> {
    let a = input(ins, 0, "Resize")?;
    if n.str("mode").unwrap_or("nearest") != "nearest" {
        bail!("Resize: only mode nearest is supported");
    }
    let rank = a.shape.len();
    if rank != 4 {
        bail!("Resize: 4-D tensors only");
    }
    let (scales, sizes) = (ins.get(2).and_then(Arg::get), ins.get(3).and_then(Arg::get));
    let (out_shape, sc): (Vec<usize>, Vec<f32>) = match (scales, sizes) {
        (Some(s), _) if !s.is_empty() => {
            if s.len() != rank {
                bail!("Resize: {} scales for rank {rank}", s.len());
            }
            (
                a.shape
                    .iter()
                    .zip(&s.data)
                    .map(|(&d, &k)| (d as f32 * k).floor() as usize)
                    .collect(),
                s.data.clone(),
            )
        }
        (_, Some(sz)) if sz.len() == rank => {
            let o: Vec<usize> = sz.ints().iter().map(|&v| v as usize).collect();
            let sc = o.iter().zip(&a.shape).map(|(&o, &i)| o as f32 / i as f32).collect();
            (o, sc)
        }
        _ => bail!("Resize without scales or sizes"),
    };
    let (nb, c, h, w) = (a.shape[0], a.shape[1], a.shape[2], a.shape[3]);
    let (oh, ow) = (out_shape[2], out_shape[3]);
    if out_shape[0] != nb || out_shape[1] != c {
        bail!("Resize: only spatial resizing is supported");
    }
    let ctm = n.str("coordinate_transformation_mode").unwrap_or("half_pixel");
    let nm = n.str("nearest_mode").unwrap_or("round_prefer_floor");
    let src = |o: usize, scale: f32, len: usize, olen: usize| -> Result<usize> {
        let x = match ctm {
            "asymmetric" => o as f32 / scale,
            "half_pixel" => (o as f32 + 0.5) / scale - 0.5,
            "pytorch_half_pixel" if olen > 1 => (o as f32 + 0.5) / scale - 0.5,
            "pytorch_half_pixel" => 0.0,
            "align_corners" if olen > 1 => o as f32 * (len as f32 - 1.0) / (olen as f32 - 1.0),
            "align_corners" => 0.0,
            m => bail!("Resize: coordinate_transformation_mode {m} is not supported"),
        };
        let i = match nm {
            "floor" => x.floor(),
            "ceil" => x.ceil(),
            "round_prefer_ceil" => (x + 0.5).floor(),
            "round_prefer_floor" if x.fract() == 0.5 => x.floor(),
            "round_prefer_floor" => x.round(),
            m => bail!("Resize: nearest_mode {m} is not supported"),
        };
        Ok((i.max(0.0) as usize).min(len - 1))
    };
    let ys: Vec<usize> = (0..oh).map(|y| src(y, sc[2], h, oh)).collect::<Result<_>>()?;
    let xs: Vec<usize> = (0..ow).map(|x| src(x, sc[3], w, ow)).collect::<Result<_>>()?;
    let mut data = vec![0f32; nb * c * oh * ow];
    if a.is_cl() {
        let par = nb * oh > 1 && data.len() > 65536;
        crate::kernels::for_chunks(&mut data, ow * c, par, |r, out| {
            let (i, y) = (r / oh, r % oh);
            let row = &a.data[(i * h + ys[y]) * w * c..(i * h + ys[y] + 1) * w * c];
            for (x, &sx) in xs.iter().enumerate() {
                out[x * c..(x + 1) * c].copy_from_slice(&row[sx * c..(sx + 1) * c]);
            }
        });
        return Ok(Tensor::new(out_shape, data).with_layout(true));
    }
    let par = nb * c > 1 && data.len() > 65536;
    crate::kernels::for_chunks(&mut data, oh * ow, par, |plane, out| {
        let s = &a.data[plane * h * w..(plane + 1) * h * w];
        for (y, &sy) in ys.iter().enumerate() {
            for (x, &sx) in xs.iter().enumerate() {
                out[y * ow + x] = s[sy * w + sx];
            }
        }
    });
    Ok(Tensor::new(out_shape, data))
}

// ---------------------------------------------------------------------------------------------
// Building models (tests and tools).

/// Writes a small ONNX model: for tests, which build graphs in code.
#[doc(hidden)]
pub mod build {
    fn varint(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }

    fn key(out: &mut Vec<u8>, num: u32, wt: u8) {
        varint(out, ((num as u64) << 3) | wt as u64);
    }

    fn bytes(out: &mut Vec<u8>, num: u32, b: &[u8]) {
        key(out, num, 2);
        varint(out, b.len() as u64);
        out.extend_from_slice(b);
    }

    fn int(out: &mut Vec<u8>, num: u32, v: i64) {
        key(out, num, 0);
        varint(out, v as u64);
    }

    /// A tensor: `dtype` 1 (f32 values) or an integer type (`ints`), as raw data.
    pub fn tensor(name: &str, dims: &[usize], dtype: i64, f: &[f32], ints: &[i64]) -> Vec<u8> {
        let mut t = Vec::new();
        for &d in dims {
            int(&mut t, 1, d as i64);
        }
        int(&mut t, 2, dtype);
        bytes(&mut t, 8, name.as_bytes());
        let raw: Vec<u8> = match dtype {
            1 => f.iter().flat_map(|v| v.to_le_bytes()).collect(),
            2 => ints.iter().map(|&v| v as u8).collect(),
            3 => ints.iter().map(|&v| v as i8 as u8).collect(),
            6 => ints.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect(),
            _ => ints.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        bytes(&mut t, 9, &raw);
        t
    }

    /// A node attribute.
    pub enum A<'a> {
        I(i64),
        F(f32),
        S(&'a str),
        Is(&'a [i64]),
        T(Vec<u8>),
    }

    pub fn node(op: &str, ins: &[&str], outs: &[&str], attrs: &[(&str, A)]) -> Vec<u8> {
        let mut n = Vec::new();
        for i in ins {
            bytes(&mut n, 1, i.as_bytes());
        }
        for o in outs {
            bytes(&mut n, 2, o.as_bytes());
        }
        bytes(&mut n, 4, op.as_bytes());
        for (k, a) in attrs {
            let mut at = Vec::new();
            bytes(&mut at, 1, k.as_bytes());
            match a {
                A::I(v) => {
                    int(&mut at, 3, *v);
                    int(&mut at, 20, 2);
                }
                A::F(v) => {
                    key(&mut at, 2, 5);
                    at.extend_from_slice(&v.to_le_bytes());
                    int(&mut at, 20, 1);
                }
                A::S(s) => {
                    bytes(&mut at, 4, s.as_bytes());
                    int(&mut at, 20, 3);
                }
                A::Is(v) => {
                    let mut p = Vec::new();
                    for &x in v.iter() {
                        varint(&mut p, x as u64);
                    }
                    bytes(&mut at, 8, &p);
                    int(&mut at, 20, 7);
                }
                A::T(t) => {
                    bytes(&mut at, 5, t);
                    int(&mut at, 20, 4);
                }
            }
            bytes(&mut n, 5, &at);
        }
        n
    }

    /// A model of `nodes` with initializers `inits` (from [`tensor`]), graph inputs and outputs.
    pub fn model(nodes: &[Vec<u8>], inits: &[Vec<u8>], inputs: &[&str], outputs: &[&str]) -> Vec<u8> {
        let mut g = Vec::new();
        for n in nodes {
            bytes(&mut g, 1, n);
        }
        bytes(&mut g, 2, b"g");
        for t in inits {
            bytes(&mut g, 5, t);
        }
        let vi = |name: &str| {
            let mut v = Vec::new();
            bytes(&mut v, 1, name.as_bytes());
            v
        };
        for i in inputs {
            bytes(&mut g, 11, &vi(i));
        }
        for o in outputs {
            bytes(&mut g, 12, &vi(o));
        }
        let mut m = Vec::new();
        int(&mut m, 1, 8);
        let mut os = Vec::new();
        bytes(&mut os, 1, b"");
        int(&mut os, 2, 17);
        bytes(&mut m, 8, &os);
        bytes(&mut m, 7, &g);
        m
    }
}

#[cfg(test)]
mod tests {
    use super::build::*;
    use super::*;

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Plain NCHW reference convolution.
    #[allow(clippy::too_many_arguments)]
    fn conv_ref(
        x: &[f32],
        (nb, cin, h, w): (usize, usize, usize, usize),
        wt: &[f32],
        b: &[f32],
        (cout, kh, kw): (usize, usize, usize),
        group: usize,
        st: [usize; 2],
        pads: [usize; 4],
    ) -> (Vec<f32>, usize, usize) {
        let cpg = cin / group;
        let copg = cout / group;
        let oh = (h + pads[0] + pads[2] - kh) / st[0] + 1;
        let ow = (w + pads[1] + pads[3] - kw) / st[1] + 1;
        let mut y = vec![0f32; nb * cout * oh * ow];
        for i in 0..nb {
            for co in 0..cout {
                let g = co / copg;
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut s = b[co];
                        for ci in 0..cpg {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = (oy * st[0] + ky) as isize - pads[0] as isize;
                                    let ix = (ox * st[1] + kx) as isize - pads[1] as isize;
                                    if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                                        s += x[((i * cin + g * cpg + ci) * h + iy as usize) * w + ix as usize]
                                            * wt[((co * cpg + ci) * kh + ky) * kw + kx];
                                    }
                                }
                            }
                        }
                        y[((i * cout + co) * oh + oy) * ow + ox] = s;
                    }
                }
            }
        }
        (y, oh, ow)
    }

    #[test]
    fn protobuf_varints() {
        let mut p = Pb::new(&[0x96, 0x01, 0x08]);
        assert_eq!(p.varint().unwrap(), 150);
        assert_eq!(p.varint().unwrap(), 8);
        assert!(p.varint().is_err());
        assert!(Model::parse(&[0x3a, 0x05, 0x01]).is_err(), "truncated bytes field");
    }

    #[test]
    fn convs_match_reference() {
        // (cin, cout, k, group, stride, pad): pointwise, depthwise s1 / s2, a stem, grouped.
        let cases = [
            (8, 12, 1, 1, [1, 1], 0),
            (8, 8, 3, 8, [1, 1], 1),
            (8, 8, 3, 8, [2, 2], 1),
            (8, 8, 3, 8, [2, 1], 1),
            (1, 6, 3, 1, [2, 2], 1),
            (4, 6, 3, 2, [1, 1], 1),
        ];
        for (ci, &(cin, cout, k, group, st, pad)) in cases.iter().enumerate() {
            let (nb, h, w) = (2, 7, 9);
            let x = rnd(nb * cin * h * w, ci as u64);
            let wt = rnd(cout * (cin / group) * k * k, 100 + ci as u64);
            let b = rnd(cout, 200 + ci as u64);
            let m = model(
                &[
                    node(
                        "Conv",
                        &["x", "w", "b"],
                        &["c"],
                        &[
                            ("group", A::I(group as i64)),
                            ("kernel_shape", A::Is(&[k as i64, k as i64])),
                            ("pads", A::Is(&[pad, pad, pad, pad])),
                            ("strides", A::Is(&[st[0] as i64, st[1] as i64])),
                        ],
                    ),
                    node("Relu", &["c"], &["y"], &[]),
                ],
                &[
                    tensor("w", &[cout, cin / group, k, k], 1, &wt, &[]),
                    tensor("b", &[cout], 1, &b, &[]),
                ],
                &["x"],
                &["y"],
            );
            let m = Model::parse(&m).unwrap();
            assert_eq!(m.ops(), ["Conv"], "Relu fused");
            let y = m.run(vec![("x", Tensor::new(vec![nb, cin, h, w], x.clone()))]).unwrap().remove(0);
            let p = pad as usize;
            let (r, oh, ow) = conv_ref(&x, (nb, cin, h, w), &wt, &b, (cout, k, k), group, st, [p; 4]);
            assert_eq!(y.shape, [nb, cout, oh, ow]);
            for (a, b) in y.data.iter().zip(&r) {
                assert!((a - b.max(0.0)).abs() < 1e-5, "case {ci}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn conv1d_squeeze_and_ctc_head() {
        // x [1, 2, 1, 5] -> Squeeze(2) -> depthwise k3 (1D) -> ArgMax / conf over channels.
        let x = rnd(10, 7);
        let w = rnd(2 * 3, 8);
        let m = model(
            &[
                node("Constant", &[], &["ax"], &[("value", A::T(tensor("v", &[1], 7, &[], &[2])))]),
                node("Conv", &["x", "pw"], &["c0"], &[]),
                node("Squeeze", &["c0", "ax"], &["s"], &[]),
                node(
                    "Conv",
                    &["s", "w"],
                    &["h"],
                    &[("group", A::I(2)), ("pads", A::Is(&[1, 1])), ("kernel_shape", A::Is(&[3]))],
                ),
                node("ArgMax", &["h"], &["ids"], &[("axis", A::I(1)), ("keepdims", A::I(0))]),
                node("ReduceMax", &["h"], &["mx"], &[("axes", A::Is(&[1])), ("keepdims", A::I(0))]),
                node("ReduceLogSumExp", &["h"], &["lse"], &[("axes", A::Is(&[1])), ("keepdims", A::I(0))]),
                node("Sub", &["mx", "lse"], &["d"], &[]),
                node("Exp", &["d"], &["conf"], &[]),
            ],
            &[
                tensor("w", &[2, 1, 3], 1, &w, &[]),
                tensor("pw", &[2, 2, 1, 1], 1, &[1.0, 0.0, 0.0, 1.0], &[]),
            ],
            &["x"],
            &["ids", "conf"],
        );
        let m = Model::parse(&m).unwrap();
        let out = m.run(vec![("x", Tensor::new(vec![1, 2, 1, 5], x.clone()))]).unwrap();
        assert_eq!(out[0].shape, [1, 5]);
        for t in 0..5 {
            let h: Vec<f32> = (0..2)
                .map(|c| {
                    (0..3)
                        .map(|k| {
                            let i = t as isize + k as isize - 1;
                            if (0..5).contains(&i) {
                                x[c * 5 + i as usize] * w[c * 3 + k]
                            } else {
                                0.0
                            }
                        })
                        .sum()
                })
                .collect();
            let id = if h[1] > h[0] { 1 } else { 0 };
            assert_eq!(out[0].ints.as_ref().unwrap()[t], id);
            let conf = 1.0 / (1.0 + (h[1 - id as usize] - h[id as usize]).exp());
            assert!((out[1].data[t] - conf).abs() < 1e-5);
        }
    }

    #[test]
    fn detector_head_ops() {
        // Mul by a broadcast constant, ReduceSum over channels, Resize x2 nearest, Concat,
        // Sigmoid; and Gemm with transB after a ReduceMean.
        let x = rnd(3 * 2 * 2, 3);
        let m = model(
            &[
                node("Mul", &["x", "k"], &["m"], &[]),
                node("ReduceSum", &["m", "ax"], &["g"], &[("keepdims", A::I(1))]),
                node(
                    "Resize",
                    &["g", "", "sc"],
                    &["r"],
                    &[
                        ("mode", A::S("nearest")),
                        ("coordinate_transformation_mode", A::S("asymmetric")),
                        ("nearest_mode", A::S("floor")),
                    ],
                ),
                node("Concat", &["r", "r"], &["c"], &[("axis", A::I(1))]),
                node("Sigmoid", &["c"], &["y"], &[]),
                node("ReduceMean", &["x"], &["p"], &[("axes", A::Is(&[2, 3])), ("keepdims", A::I(0))]),
                node("Gemm", &["p", "gw", "gb"], &["z"], &[("transB", A::I(1))]),
            ],
            &[
                tensor("k", &[1, 3, 1, 1], 1, &[0.299, 0.587, 0.114], &[]),
                tensor("ax", &[1], 7, &[], &[1]),
                tensor("sc", &[4], 1, &[1.0, 1.0, 2.0, 2.0], &[]),
                tensor("gw", &[2, 3], 1, &[1.0, 0.0, 0.0, 0.0, 1.0, 1.0], &[]),
                tensor("gb", &[2], 1, &[0.5, -0.5], &[]),
            ],
            &["x"],
            &["y", "z"],
        );
        let m = Model::parse(&m).unwrap();
        let out = m.run(vec![("x", Tensor::new(vec![1, 3, 2, 2], x.clone()))]).unwrap();
        assert_eq!(out[0].shape, [1, 2, 4, 4]);
        for c in 0..2 {
            for y in 0..4 {
                for xx in 0..4 {
                    let p = (y / 2) * 2 + xx / 2;
                    let g = 0.299 * x[p] + 0.587 * x[4 + p] + 0.114 * x[8 + p];
                    let v = out[0].data[(c * 4 + y) * 4 + xx];
                    assert!((v - 1.0 / (1.0 + (-g).exp())).abs() < 1e-6);
                }
            }
        }
        let mean = |c: usize| x[c * 4..c * 4 + 4].iter().sum::<f32>() / 4.0;
        assert_eq!(out[1].shape, [1, 2]);
        assert!((out[1].data[0] - (mean(0) + 0.5)).abs() < 1e-6);
        assert!((out[1].data[1] - (mean(1) + mean(2) - 0.5)).abs() < 1e-6);
    }

    #[test]
    fn qdq_fake_quantization() {
        // Weights int8 per channel through DequantizeLinear (folded), the input quantized to
        // uint8 and back, the conv output too (fused into the Conv).
        let wq = [10i64, -20, 30, -40];
        let ws = [0.01f32, 0.02];
        let m = model(
            &[
                node("DequantizeLinear", &["wq", "ws", "wz"], &["w"], &[("axis", A::I(0))]),
                node("QuantizeLinear", &["x", "s", "z"], &["xq"], &[]),
                node("DequantizeLinear", &["xq", "s", "z"], &["xd"], &[]),
                node("Conv", &["xd", "w"], &["c"], &[]),
                node("QuantizeLinear", &["c", "s2", "z2"], &["cq"], &[]),
                node("DequantizeLinear", &["cq", "s2", "z2"], &["y"], &[]),
            ],
            &[
                tensor("wq", &[2, 2, 1, 1], 3, &[], &wq),
                tensor("ws", &[2], 1, &ws, &[]),
                tensor("wz", &[2], 3, &[], &[0, 0]),
                tensor("s", &[], 1, &[1.0 / 127.5], &[]),
                tensor("z", &[], 2, &[], &[128]),
                tensor("s2", &[], 1, &[0.05], &[]),
                tensor("z2", &[], 2, &[], &[0]),
            ],
            &["x"],
            &["y"],
        );
        let m = Model::parse(&m).unwrap();
        assert_eq!(m.ops(), ["FakeQuant", "Conv"]);
        // Only the dequantized weights stay.
        assert_eq!(m.params(), 4);
        let x = vec![0.3, -0.7, 0.9, 0.1, -1.0, 0.25];
        let y = m.run(vec![("x", Tensor::new(vec![1, 2, 1, 3], x.clone()))]).unwrap().remove(0);
        let fq = |v: f32, s: f32, z: f32| (((v / s).round_ties_even() + z).clamp(0.0, 255.0) - z) * s;
        for p in 0..3 {
            let (a, b) = (fq(x[p], 1.0 / 127.5, 128.0), fq(x[3 + p], 1.0 / 127.5, 128.0));
            for co in 0..2 {
                let c = a * wq[co * 2] as f32 * ws[co] + b * wq[co * 2 + 1] as f32 * ws[co];
                assert!((y.data[co * 3 + p] - fq(c, 0.05, 0.0)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn missing_values_are_reported() {
        let m = model(&[node("Relu", &["nope"], &["y"], &[])], &[], &["x"], &["y"]);
        assert!(Model::parse(&m).is_err());
        let m = model(&[node("Foo", &["x"], &["y"], &[])], &[], &["x"], &["y"]);
        let m = Model::parse(&m).unwrap();
        assert!(m.run(vec![("x", Tensor::new(vec![1], vec![0.0]))]).is_err());
    }

    /// A `.npy` file (little-endian f4 / i8, C order) as a tensor.
    pub(crate) fn npy(path: &std::path::Path) -> Tensor {
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(&b[..6], b"\x93NUMPY");
        let (hl, off) = if b[6] == 1 {
            (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
        } else {
            (u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize, 12)
        };
        let h = std::str::from_utf8(&b[off..off + hl]).unwrap();
        let shape_s = &h[h.find("'shape': (").unwrap() + 10..];
        let shape: Vec<usize> = shape_s[..shape_s.find(')').unwrap()]
            .split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect();
        let d = &b[off + hl..];
        if h.contains("'<i8'") {
            Tensor::from_ints(shape, d.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect())
        } else {
            assert!(h.contains("'<f4'"), "{h}");
            Tensor::new(shape, d.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
        }
    }

    /// The models against onnxruntime's outputs dumped by the reference pipeline
    /// (`OXINFER_OCRIA_MODELS`: the model folder, `OXINFER_OCRIA_DUMPS`: one folder per image).
    #[test]
    #[ignore]
    fn real_models_match_onnxruntime() {
        let (Some(md), Some(dd)) = (std::env::var_os("OXINFER_OCRIA_MODELS"), std::env::var_os("OXINFER_OCRIA_DUMPS")) else {
            return;
        };
        let (md, dd) = (std::path::PathBuf::from(md), std::path::PathBuf::from(dd));
        let det = Model::load(&md.join("det.onnx")).unwrap();
        let router = Model::load(&md.join("router.onnx")).unwrap();
        let recs: Vec<(String, Model)> = ["latin", "cjk", "rtl", "world"]
            .iter()
            .map(|g| (g.to_string(), Model::load(&md.join(format!("rec_{g}.onnx"))).unwrap()))
            .collect();
        let mut dirs: Vec<_> = std::fs::read_dir(&dd)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.join("det_input.npy").exists())
            .collect();
        dirs.sort();
        let maxdiff = |a: &Tensor, b: &Tensor| {
            assert_eq!(a.shape, b.shape);
            a.data.iter().zip(&b.data).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
        };
        for d in dirs {
            let name = d.file_name().unwrap().to_string_lossy().to_string();
            let x = npy(&d.join("det_input.npy"));
            let t0 = std::time::Instant::now();
            let p = det.run(vec![("x", x.clone())]).unwrap().remove(0);
            let dt = t0.elapsed();
            let pr = npy(&d.join("det_prob.npy"));
            let flips = p.data.iter().zip(&pr.data).filter(|(a, b)| (**a > 0.55) != (**b > 0.55)).count();
            eprintln!(
                "{name}: det {:?} max diff {:.2e}, {flips} cells across the threshold, {:.1} ms",
                x.shape,
                maxdiff(&p, &pr),
                dt.as_secs_f64() * 1e3
            );
            if d.join("router_input.npy").exists() {
                let x = npy(&d.join("router_input.npy"));
                let out = router.run(vec![("x", x)]).unwrap();
                let (g, f) = (npy(&d.join("router_group_logits.npy")), npy(&d.join("router_flip_logits.npy")));
                eprintln!(
                    "{name}: router max diff group {:.2e} flip {:.2e}",
                    maxdiff(&out[0], &g),
                    maxdiff(&out[1], &f)
                );
            }
            for (g, m) in &recs {
                for b in 0.. {
                    let f = d.join(format!("rec_{g}_b{b}_input.npy"));
                    if !f.exists() {
                        break;
                    }
                    let x = npy(&f);
                    let t0 = std::time::Instant::now();
                    let out = m.run(vec![("x", x.clone())]).unwrap();
                    let dt = t0.elapsed();
                    let (ids, conf) = (
                        npy(&d.join(format!("rec_{g}_b{b}_ids.npy"))),
                        npy(&d.join(format!("rec_{g}_b{b}_conf.npy"))),
                    );
                    assert_eq!(out[0].shape, ids.shape);
                    let bad = out[0]
                        .ints
                        .as_ref()
                        .unwrap()
                        .iter()
                        .zip(ids.ints.as_ref().unwrap())
                        .filter(|(a, b)| a != b)
                        .count();
                    eprintln!(
                        "{name}: rec_{g} batch {b} {:?}: {bad} of {} ids differ, conf max diff {:.2e}, {:.1} ms",
                        x.shape,
                        ids.len(),
                        maxdiff(&out[1], &conf),
                        dt.as_secs_f64() * 1e3
                    );
                    // A quantized graph against onnxruntime without its optimizations (the QDQ
                    // nodes run in float, as here).
                    let f = d.join(format!("rec_{g}_b{b}_conf_noopt.npy"));
                    if f.exists() {
                        let (ids, conf) = (npy(&d.join(format!("rec_{g}_b{b}_ids_noopt.npy"))), npy(&f));
                        let bad = out[0]
                            .ints
                            .as_ref()
                            .unwrap()
                            .iter()
                            .zip(ids.ints.as_ref().unwrap())
                            .filter(|(a, b)| a != b)
                            .count();
                        eprintln!(
                            "{name}: rec_{g} batch {b} against float QDQ: {bad} ids differ, conf max diff {:.2e}",
                            maxdiff(&out[1], &conf)
                        );
                    }
                }
            }
        }
    }
}

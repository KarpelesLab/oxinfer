//! The ONNX interpreter's operators on an accelerator ([`crate::onnx`]): the graph's constants
//! (convolution weights, biases, requantization multipliers) uploaded once per model
//! ([`Device::constant`]), a run's tensors in device buffers of a [`Session`] (channels-last,
//! f32, readable and writable by the host between runs: unified memory), and the operators as
//! [`Step`]s, each run in order. The interpreter runs the nodes it can on the device and the
//! others on the CPU, moving tensors where they are needed.

/// The steps after an operator, in this order: ReLU, fake quantization `(clamp(round(v / s) +
/// zp, lo, hi) - zp2) * s2` (`fq` = `[s, zp, lo, hi, s2, zp2]`), the end of an integer step
/// `(clamp(round(v * mult[c]) + zp, lo, hi) - zp2) * s2` (`requant` = the multipliers' constant
/// and `[zp, lo, hi, s2, zp2]`), the logistic function, the rounding of an integer sum.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Post {
    pub relu: bool,
    pub fq: Option<[f32; 6]>,
    pub requant: Option<(usize, [f32; 5])>,
    pub sigmoid: bool,
    /// `(clamp(round(v) + zp, lo, hi) - zp2) * s2` (`[zp, lo, hi, s2, zp2]`), last.
    pub round: Option<[f32; 5]>,
}

/// A convolution's geometry (1D convolutions as 2D with `h = 1`), channels-last tensors,
/// weights `[cout][cin / groups][kh][kw]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvGeom {
    pub n: usize,
    pub h: usize,
    pub w: usize,
    pub cin: usize,
    pub oh: usize,
    pub ow: usize,
    pub cout: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: [usize; 2],
    pub dil: [usize; 2],
    /// Top, left.
    pub pad: [usize; 2],
    pub groups: usize,
}

/// One operator. Buffers are a [`Session`]'s, constants the [`Device`]'s.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// Any convolution, one thread per output value; `s_in > 0`: the input holds a
    /// fake-quantized tensor whose integers (`round(x / s_in)`) are convolved.
    Conv {
        x: usize,
        y: usize,
        w: usize,
        b: usize,
        g: ConvGeom,
        s_in: f32,
        post: Post,
    },
    /// A pointwise convolution over every input channel as a GEMM (`cin % 8 == 0`, the weights
    /// `[cout padded to 32][cin]`; the input buffer has 32 rows of room past `npix`).
    Pointwise {
        x: usize,
        y: usize,
        w: usize,
        b: usize,
        npix: usize,
        cin: usize,
        cout: usize,
        post: Post,
    },
    /// `y = round(x / s)` (the integers behind a fake-quantized tensor).
    Round { x: usize, y: usize, n: usize, s: f32 },
    /// `y = post(a + b)`, or with `exact = [sa, sb, rac, rbc, zc]` ONNX Runtime's `QLinearAdd`
    /// sum `rac round(a / sa) + rbc round(b / sb) + zc`.
    Add {
        a: usize,
        b: usize,
        y: usize,
        n: usize,
        exact: Option<[f32; 5]>,
        post: Post,
    },
    /// `y = post(x)`.
    Unary { x: usize, y: usize, n: usize, post: Post },
    /// Nearest resize: `y[i][oy][ox][c] = x[i][ys[oy]][xs[ox]][c]`.
    Resize {
        x: usize,
        y: usize,
        ys: Vec<u32>,
        xs: Vec<u32>,
        n: usize,
        h: usize,
        w: usize,
        c: usize,
    },
    /// `dst[p][off + c] = src[p][c]` for `c < cs` (`dst` rows `cd` wide): a part of a channel
    /// concatenation.
    Cat {
        src: usize,
        dst: usize,
        npix: usize,
        cs: usize,
        cd: usize,
        off: usize,
    },
}

/// A device's copy of a model's constants.
pub trait Device: Send + Sync {
    /// Uploads `v`; its id for [`Step`]s.
    fn constant(&self, v: &[f32]) -> Option<usize>;
    /// A run's buffers.
    fn session(&self) -> Option<Box<dyn Session + '_>>;
}

/// A run's device buffers.
pub trait Session {
    /// A buffer of `floats` values (and room for 32 rows of up to 1024 values past them).
    fn alloc(&mut self, floats: usize) -> Option<usize>;
    /// Gives a buffer back for reuse.
    fn release(&mut self, buf: usize);
    /// The buffer's values, for the host (between [`Session::run`]s).
    fn data(&mut self, buf: usize) -> &mut [f32];
    /// Runs the steps in order; false when the device failed.
    fn run(&mut self, steps: &[Step]) -> bool;
}

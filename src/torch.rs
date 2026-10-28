//! Reader for PyTorch `torch.save` checkpoints (`pytorch_model.bin`, `*.pt`): a zip archive whose
//! entries are stored uncompressed, holding a pickled state dict (`<prefix>/data.pkl`) whose
//! tensors point at raw storage files (`<prefix>/data/<key>`). Nothing is unpickled beyond what
//! state dicts use, and no Python code is ever run: unknown callables are just opaque values.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;

/// A tensor of the checkpoint: its dtype (safetensors spelling), shape and absolute byte range.
#[derive(Debug, Clone)]
pub struct TorchTensor {
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    /// Element strides (row-major for [`read_tensors`]; any dense permutation, e.g. a stored
    /// transpose, for [`read_tensors_dense`]).
    pub stride: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

/// True when `bytes` starts like a zip archive (as `torch.save` writes them).
pub fn is_torch_zip(bytes: &[u8]) -> bool {
    bytes.starts_with(b"PK\x03\x04")
}

fn u16_at(b: &[u8], o: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(b.get(o..o + 2).context("zip: truncated")?.try_into().unwrap()))
}
fn u32_at(b: &[u8], o: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(b.get(o..o + 4).context("zip: truncated")?.try_into().unwrap()))
}
fn u64_at(b: &[u8], o: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(b.get(o..o + 8).context("zip: truncated")?.try_into().unwrap()))
}

/// The archive's entries: name -> (data offset, size). Only stored (uncompressed) entries.
fn zip_entries(b: &[u8]) -> Result<HashMap<String, (usize, usize)>> {
    // End of central directory: the last record, followed by a comment of at most 64 KiB.
    let lo = b.len().saturating_sub(22 + 65535);
    let eocd = (lo..b.len().saturating_sub(21))
        .rev()
        .find(|&i| b[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .context("zip: no end of central directory")?;
    let (mut count, mut cd_off) = (u16_at(b, eocd + 10)? as u64, u32_at(b, eocd + 16)? as u64);
    if count == 0xffff || cd_off == 0xffff_ffff {
        // Zip64: the locator sits right before the end record and points at the zip64 record.
        let loc = eocd.checked_sub(20).context("zip64: no locator")?;
        if b[loc..loc + 4] != [0x50, 0x4b, 0x06, 0x07] {
            bail!("zip64: bad locator");
        }
        let rec = u64_at(b, loc + 8)? as usize;
        if b.get(rec..rec + 4) != Some(&[0x50, 0x4b, 0x06, 0x06]) {
            bail!("zip64: bad end record");
        }
        count = u64_at(b, rec + 32)?;
        cd_off = u64_at(b, rec + 48)?;
    }
    let mut out = HashMap::new();
    let mut p = cd_off as usize;
    for _ in 0..count {
        if u32_at(b, p)? != 0x0201_4b50 {
            bail!("zip: bad central directory entry");
        }
        let method = u16_at(b, p + 10)?;
        let (mut csize, mut usize_) = (u32_at(b, p + 20)? as u64, u32_at(b, p + 24)? as u64);
        let (nlen, xlen, clen) = (
            u16_at(b, p + 28)? as usize,
            u16_at(b, p + 30)? as usize,
            u16_at(b, p + 32)? as usize,
        );
        let mut local = u32_at(b, p + 42)? as u64;
        let name = String::from_utf8_lossy(b.get(p + 46..p + 46 + nlen).context("zip: truncated")?).into_owned();
        // Zip64 extra field: the 32-bit fields that overflowed, in this order.
        let mut x = p + 46 + nlen;
        let xend = x + xlen;
        while x + 4 <= xend {
            let (id, sz) = (u16_at(b, x)?, u16_at(b, x + 2)? as usize);
            if id == 1 {
                let mut q = x + 4;
                if usize_ == 0xffff_ffff {
                    usize_ = u64_at(b, q)?;
                    q += 8;
                }
                if csize == 0xffff_ffff {
                    csize = u64_at(b, q)?;
                    q += 8;
                }
                if local == 0xffff_ffff {
                    local = u64_at(b, q)?;
                }
            }
            x += 4 + sz;
        }
        p = xend + clen;
        if method != 0 {
            continue; // compressed: never the case for tensor data written by torch.save
        }
        let l = local as usize;
        if u32_at(b, l)? != 0x0403_4b50 {
            bail!("zip: bad local header for {name}");
        }
        let data = l + 30 + u16_at(b, l + 26)? as usize + u16_at(b, l + 28)? as usize;
        if data + csize as usize > b.len() {
            bail!("zip: {name} out of bounds");
        }
        let _ = usize_;
        out.insert(name, (data, csize as usize));
    }
    Ok(out)
}

#[derive(Debug, Clone)]
enum V {
    None,
    Bool,
    Int(i64),
    Float,
    Str(String),
    Tuple(Vec<V>),
    List(Vec<V>),
    Dict(Vec<(V, V)>),
    Global(String),
    Storage(String, &'static str),
    Tensor {
        key: String,
        dtype: &'static str,
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<usize>,
    },
    Mark,
    Opaque,
}

fn storage_dtype(class: &str) -> Option<&'static str> {
    Some(match class.rsplit('.').next()? {
        "FloatStorage" => "F32",
        "HalfStorage" => "F16",
        "BFloat16Storage" => "BF16",
        "DoubleStorage" => "F64",
        "LongStorage" => "I64",
        "IntStorage" => "I32",
        "ShortStorage" => "I16",
        "CharStorage" => "I8",
        "ByteStorage" => "U8",
        "BoolStorage" => "BOOL",
        _ => return None,
    })
}

fn dtype_size(d: &str) -> usize {
    match d {
        "F64" | "I64" => 8,
        "F32" | "I32" => 4,
        "F16" | "BF16" | "I16" => 2,
        _ => 1,
    }
}

fn ints(v: &V) -> Option<Vec<usize>> {
    match v {
        V::Tuple(t) | V::List(t) => t.iter().map(|x| if let V::Int(i) = x { Some(*i as usize) } else { None }).collect(),
        _ => None,
    }
}

/// Runs the pickle program and returns the unpickled object.
fn unpickle(p: &[u8]) -> Result<V> {
    let mut stack: Vec<V> = Vec::new();
    let mut memo: HashMap<u32, V> = HashMap::new();
    let mut i = 0usize;
    let take = |i: &mut usize, n: usize| -> Result<&[u8]> {
        let s = p.get(*i..*i + n).context("pickle: truncated")?;
        *i += n;
        Ok(s)
    };
    let pop_mark = |stack: &mut Vec<V>| -> Result<Vec<V>> {
        let m = stack.iter().rposition(|v| matches!(v, V::Mark)).context("pickle: no mark")?;
        let items = stack.split_off(m + 1);
        stack.pop();
        Ok(items)
    };
    loop {
        let op = *p.get(i).context("pickle: no STOP")?;
        i += 1;
        match op {
            0x80 => i += 1,                                            // PROTO
            0x95 => i += 8,                                            // FRAME
            b'.' => return stack.pop().context("pickle: empty stack"), // STOP
            b'}' => stack.push(V::Dict(Vec::new())),
            b']' => stack.push(V::List(Vec::new())),
            b')' => stack.push(V::Tuple(Vec::new())),
            0x8f => stack.push(V::List(Vec::new())), // EMPTY_SET
            b'(' => stack.push(V::Mark),
            b'N' => stack.push(V::None),
            0x88 | 0x89 => stack.push(V::Bool),
            b'J' => stack.push(V::Int(i32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap()) as i64)),
            b'K' => stack.push(V::Int(take(&mut i, 1)?[0] as i64)),
            b'M' => stack.push(V::Int(u16::from_le_bytes(take(&mut i, 2)?.try_into().unwrap()) as i64)),
            0x8a => {
                let n = take(&mut i, 1)?[0] as usize;
                let b = take(&mut i, n)?;
                let mut v: i64 = 0;
                for (k, &x) in b.iter().enumerate().take(8) {
                    v |= (x as i64) << (8 * k);
                }
                if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                stack.push(V::Int(v));
            }
            b'G' => {
                take(&mut i, 8)?;
                stack.push(V::Float);
            }
            b'X' | 0x8d | 0x8c | b'T' | b'U' | b'B' | b'C' | 0x8e => {
                let n = match op {
                    b'X' | b'T' | b'B' => u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap()) as usize,
                    0x8d | 0x8e => u64::from_le_bytes(take(&mut i, 8)?.try_into().unwrap()) as usize,
                    _ => take(&mut i, 1)?[0] as usize,
                };
                let s = take(&mut i, n)?;
                stack.push(V::Str(String::from_utf8_lossy(s).into_owned()));
            }
            b'q' => {
                let k = take(&mut i, 1)?[0] as u32;
                memo.insert(k, stack.last().context("pickle: empty stack")?.clone());
            }
            b'r' => {
                let k = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap());
                memo.insert(k, stack.last().context("pickle: empty stack")?.clone());
            }
            0x94 => {
                let k = memo.len() as u32;
                memo.insert(k, stack.last().context("pickle: empty stack")?.clone());
            }
            b'h' => {
                let k = take(&mut i, 1)?[0] as u32;
                stack.push(memo.get(&k).context("pickle: bad memo")?.clone());
            }
            b'j' => {
                let k = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap());
                stack.push(memo.get(&k).context("pickle: bad memo")?.clone());
            }
            b'c' => {
                let rest = &p[i..];
                let a = rest.iter().position(|&c| c == b'\n').context("pickle: bad GLOBAL")?;
                let b = rest[a + 1..].iter().position(|&c| c == b'\n').context("pickle: bad GLOBAL")?;
                let module = String::from_utf8_lossy(&rest[..a]);
                let name = String::from_utf8_lossy(&rest[a + 1..a + 1 + b]);
                i += a + b + 2;
                stack.push(V::Global(format!("{module}.{name}")));
            }
            0x93 => {
                let name = stack.pop();
                let module = stack.pop();
                match (module, name) {
                    (Some(V::Str(m)), Some(V::Str(n))) => stack.push(V::Global(format!("{m}.{n}"))),
                    _ => bail!("pickle: bad STACK_GLOBAL"),
                }
            }
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(V::Tuple(items));
            }
            0x85..=0x87 => {
                let n = (op - 0x84) as usize;
                let items = stack.split_off(stack.len().checked_sub(n).context("pickle: stack underflow")?);
                stack.push(V::Tuple(items));
            }
            b'a' => {
                let v = stack.pop().context("pickle: stack underflow")?;
                if let Some(V::List(l)) = stack.last_mut() {
                    l.push(v);
                }
            }
            b'e' | 0x90 => {
                let items = pop_mark(&mut stack)?;
                if let Some(V::List(l)) = stack.last_mut() {
                    l.extend(items);
                }
            }
            b's' => {
                let v = stack.pop().context("pickle: stack underflow")?;
                let k = stack.pop().context("pickle: stack underflow")?;
                if let Some(V::Dict(d)) = stack.last_mut() {
                    d.push((k, v));
                }
            }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                if let Some(V::Dict(d)) = stack.last_mut() {
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        d.push((k, v));
                    }
                }
            }
            b'b' => {
                stack.pop(); // BUILD: the state (e.g. a state dict's `_metadata`) is not needed
            }
            b'Q' => {
                // Persistent id: ('storage', StorageClass, key, location, numel).
                let pid = stack.pop().context("pickle: stack underflow")?;
                let v = match pid {
                    V::Tuple(t) if t.len() >= 3 => match (&t[1], &t[2]) {
                        (V::Global(class), V::Str(key)) => match storage_dtype(class) {
                            Some(d) => V::Storage(key.clone(), d),
                            None => bail!("pickle: unsupported storage {class}"),
                        },
                        _ => V::Opaque,
                    },
                    _ => V::Opaque,
                };
                stack.push(v);
            }
            b'R' | 0x81 => {
                let args = stack.pop().context("pickle: stack underflow")?;
                let f = stack.pop().context("pickle: stack underflow")?;
                let args = match args {
                    V::Tuple(a) => a,
                    _ => Vec::new(),
                };
                let v = match &f {
                    V::Global(g) if g.ends_with("_rebuild_tensor_v2") || g.ends_with("._rebuild_tensor") => match args.as_slice() {
                        [V::Storage(key, dtype), V::Int(off), size, stride, ..] => V::Tensor {
                            key: key.clone(),
                            dtype,
                            offset: *off as usize,
                            shape: ints(size).context("pickle: bad tensor size")?,
                            stride: ints(stride).context("pickle: bad tensor stride")?,
                        },
                        _ => bail!("pickle: unexpected tensor arguments"),
                    },
                    V::Global(g) if g.ends_with("_rebuild_parameter") => args.into_iter().next().unwrap_or(V::Opaque),
                    V::Global(g) if g.ends_with("OrderedDict") || g == "builtins.dict" => V::Dict(Vec::new()),
                    _ => V::Opaque,
                };
                stack.push(v);
            }
            other => bail!("pickle: unsupported opcode 0x{other:02x} at {}", i - 1),
        }
    }
}

/// The tensors of a `torch.save` archive (nested dicts flatten to dotted names), with absolute
/// byte ranges into `bytes`. Tensors must be contiguous.
pub fn read_tensors(bytes: &[u8]) -> Result<Vec<(String, TorchTensor)>> {
    read_impl(bytes, false)
}

/// [`read_tensors`] that also accepts tensors stored densely in another dimension order (a
/// transposed view saved as is): their `stride` says how to index them, and `start..end` covers
/// the whole dense block.
pub fn read_tensors_dense(bytes: &[u8]) -> Result<Vec<(String, TorchTensor)>> {
    read_impl(bytes, true)
}

fn read_impl(bytes: &[u8], permuted: bool) -> Result<Vec<(String, TorchTensor)>> {
    let entries = zip_entries(bytes)?;
    let pkl = entries
        .keys()
        .find(|k| k.ends_with("/data.pkl") || *k == "data.pkl")
        .context("torch checkpoint: no data.pkl")?
        .clone();
    let prefix = pkl.strip_suffix("data.pkl").unwrap();
    let (po, pn) = entries[&pkl];
    let root = unpickle(&bytes[po..po + pn])?;
    let mut out = Vec::new();
    fn walk(
        v: &V,
        name: &str,
        prefix: &str,
        entries: &HashMap<String, (usize, usize)>,
        out: &mut Vec<(String, TorchTensor)>,
        permuted: bool,
    ) -> Result<()> {
        match v {
            V::Dict(d) => {
                for (k, v) in d {
                    if let V::Str(k) = k {
                        let n = if name.is_empty() { k.clone() } else { format!("{name}.{k}") };
                        walk(v, &n, prefix, entries, out, permuted)?;
                    }
                }
            }
            V::Tensor {
                key,
                dtype,
                offset,
                shape,
                stride,
            } => {
                let es = dtype_size(dtype);
                let mut expect = 1usize;
                let row_major = shape.iter().zip(stride).rev().all(|(d, s)| {
                    let ok = *d <= 1 || *s == expect;
                    expect *= d;
                    ok
                });
                // A dense permutation: sorted by stride, each is the product of the smaller dims.
                let dense = || {
                    let mut ds: Vec<(usize, usize)> = stride.iter().copied().zip(shape.iter().copied()).filter(|p| p.1 > 1).collect();
                    ds.sort_unstable();
                    let mut e = 1usize;
                    ds.iter().all(|&(s, d)| {
                        let ok = s == e;
                        e *= d;
                        ok
                    })
                };
                if !row_major && !(permuted && dense()) {
                    bail!("tensor {name} is not contiguous (stride {stride:?} for shape {shape:?})");
                }
                let (start, len) = *entries
                    .get(&format!("{prefix}data/{key}"))
                    .with_context(|| format!("tensor {name}: storage {key} missing"))?;
                let a = start + offset * es;
                let b = a + shape.iter().product::<usize>() * es;
                if b > start + len {
                    bail!("tensor {name} overruns its storage");
                }
                out.push((
                    name.to_string(),
                    TorchTensor {
                        dtype,
                        shape: shape.clone(),
                        stride: stride.clone(),
                        start: a,
                        end: b,
                    },
                ));
            }
            _ => {}
        }
        Ok(())
    }
    walk(&root, "", prefix, &entries, &mut out, permuted)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn reads_a_torch_save_file() {
        // torch.save({'a.weight': [[1,2,3],[4,5,6]] f32, 'b': arange(4) f16, 'c': arange(10, bf16)[2:6]})
        let bytes = include_bytes!("../tests/data/tiny.pt");
        assert!(super::is_torch_zip(bytes));
        let t: std::collections::HashMap<_, _> = super::read_tensors(bytes).unwrap().into_iter().collect();
        let a = &t["a.weight"];
        assert_eq!((a.dtype, a.shape.as_slice()), ("F32", &[2usize, 3][..]));
        let av: Vec<f32> = bytes[a.start..a.end]
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(av, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = &t["b"];
        assert_eq!((b.dtype, b.end - b.start), ("F16", 8));
        let bv: Vec<f32> = bytes[b.start..b.end]
            .chunks(2)
            .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect();
        assert_eq!(bv, [0.0, 1.0, 2.0, 3.0]);
        let c = &t["c"];
        let cv: Vec<f32> = bytes[c.start..c.end]
            .chunks(2)
            .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect();
        assert_eq!(cv, [2.0, 3.0, 4.0, 5.0]);
    }
}

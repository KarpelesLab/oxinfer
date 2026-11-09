//! Session snapshots: a model's decoding state (KV caches, Gated DeltaNet recurrent states,
//! positions, speculative decoders' caches) and the tokens it holds, written to a file and read
//! back bit for bit, so that a conversation (or a shared prompt prefix) restored later continues
//! exactly as if it had been recomputed ([`crate::engine::Engine::save_session`]); and
//! [`PromptCache`], a size-capped least-recently-used directory of such snapshots keyed by model
//! and token prefix, which [`crate::engine::Engine::chat`] restores from when a prompt starts with
//! a stored prefix.
//!
//! File layout (little-endian): the magic `OXISNAP1`, the sections' raw data one after the
//! other, a JSON footer (format version, the model fingerprint, the token count, metadata, and
//! each section's name, element type, offset and element count), the footer's length (u64), a
//! checksum of everything before it (u64) and the magic again. Readers take the sections in file
//! order and check every length against what the model expects before allocating, and the
//! checksum before the state is used.
//!
//! The cache directory is private: created with mode 0700 (an existing one is set to 0700, which
//! fails unless it is ours), files are written 0600 under a temporary name and renamed into
//! place, and symbolic links are not followed. The default directory is `prompt-cache` beside
//! the download cache ([`default_dir`]); on Linux a directory on a tmpfs gives memory-speed
//! restores.

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"OXISNAP1";
const VERSION: u64 = 1;
/// Snapshot files in a cache directory end with this.
pub const EXT: &str = "oxsnap";
/// Largest footer accepted (it holds names and offsets only).
const MAX_FOOTER: u64 = 16 << 20;

/// A checksum over 8-byte words in four independent lanes (fast enough not to slow a read from
/// memory; it detects torn or corrupted files, it is not a cryptographic hash).
#[derive(Clone)]
struct Checksum {
    lanes: [u64; 4],
    tail: Vec<u8>,
    len: u64,
}

impl Checksum {
    fn new() -> Self {
        Self {
            lanes: [
                0x9e37_79b9_7f4a_7c15,
                0xc2b2_ae3d_27d4_eb4f,
                0x1656_67b1_9e37_79f9,
                0x85eb_ca77_c2b2_ae63,
            ],
            tail: Vec::new(),
            len: 0,
        }
    }

    #[inline(always)]
    fn mix(l: u64, w: u64) -> u64 {
        (l ^ w).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(29)
    }

    fn update(&mut self, mut b: &[u8]) {
        self.len += b.len() as u64;
        if !self.tail.is_empty() {
            let need = 32 - self.tail.len();
            let take = need.min(b.len());
            self.tail.extend_from_slice(&b[..take]);
            b = &b[take..];
            if self.tail.len() < 32 {
                return;
            }
            let t = std::mem::take(&mut self.tail);
            self.block(&t);
        }
        let (chunks, rest) = b.as_chunks::<32>();
        for c in chunks {
            self.block(c);
        }
        self.tail.extend_from_slice(rest);
    }

    #[inline(always)]
    fn block(&mut self, c: &[u8]) {
        for (i, l) in self.lanes.iter_mut().enumerate() {
            let w = u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap());
            *l = Self::mix(*l, w);
        }
    }

    fn finish(&self) -> u64 {
        let mut h = self.len.wrapping_mul(0xff51_afd7_ed55_8ccd);
        for &l in &self.lanes {
            h = Self::mix(h, l);
        }
        for c in self.tail.chunks(8) {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            h = Self::mix(h, u64::from_le_bytes(w));
        }
        h ^ (h >> 31)
    }
}

/// A stable 64-bit FNV-1a hash (model fingerprints, token lists, file names).
pub fn hash64(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in parts {
        for &b in *p {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn tokens_bytes(tokens: &[u32]) -> Vec<u8> {
    tokens.iter().flat_map(|t| t.to_le_bytes()).collect()
}

/// Element types of sections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    F32,
    U32,
    U64,
}

impl Kind {
    fn size(self) -> u64 {
        match self {
            Kind::F32 | Kind::U32 => 4,
            Kind::U64 => 8,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Kind::F32 => "f32",
            Kind::U32 => "u32",
            Kind::U64 => "u64",
        }
    }
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "f32" => Kind::F32,
            "u32" => Kind::U32,
            "u64" => Kind::U64,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug)]
struct Section {
    name: String,
    kind: Kind,
    offset: u64,
    count: u64,
}

/// The bytes of a slice of plain numbers, as stored (little-endian).
fn as_bytes<T: Copy>(v: &[T]) -> std::borrow::Cow<'_, [u8]> {
    let n = std::mem::size_of_val(v);
    // SAFETY: f32 / u32 / u64 have no padding and every bit pattern is a valid byte.
    let raw = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n) };
    if cfg!(target_endian = "little") {
        std::borrow::Cow::Borrowed(raw)
    } else {
        let s = std::mem::size_of::<T>();
        std::borrow::Cow::Owned(raw.chunks(s).flat_map(|c| c.iter().rev().copied()).collect())
    }
}

/// Writes a snapshot file section by section.
pub struct SnapWriter<W: Write> {
    w: W,
    off: u64,
    sum: Checksum,
    sections: Vec<Section>,
}

impl<W: Write> SnapWriter<W> {
    pub fn new(mut w: W) -> Result<Self> {
        w.write_all(MAGIC)?;
        let mut sum = Checksum::new();
        sum.update(MAGIC);
        Ok(Self {
            w,
            off: MAGIC.len() as u64,
            sum,
            sections: Vec::new(),
        })
    }

    fn put<T: Copy>(&mut self, name: &str, kind: Kind, v: &[T]) -> Result<()> {
        if self.sections.iter().any(|s| s.name == name) {
            bail!("snapshot: section {name} written twice");
        }
        let b = as_bytes(v);
        self.sum.update(&b);
        self.w.write_all(&b)?;
        self.sections.push(Section {
            name: name.to_string(),
            kind,
            offset: self.off,
            count: v.len() as u64,
        });
        self.off += b.len() as u64;
        Ok(())
    }

    pub fn f32s(&mut self, name: &str, v: &[f32]) -> Result<()> {
        self.put(name, Kind::F32, v)
    }
    pub fn u32s(&mut self, name: &str, v: &[u32]) -> Result<()> {
        self.put(name, Kind::U32, v)
    }
    pub fn u64s(&mut self, name: &str, v: &[u64]) -> Result<()> {
        self.put(name, Kind::U64, v)
    }

    /// Writes the footer with `meta` (an object; the section table is added) and returns the
    /// writer and the file size.
    pub fn finish(mut self, mut meta: Value) -> Result<(W, u64)> {
        let secs: Vec<Value> = self
            .sections
            .iter()
            .map(|s| json!([s.name, s.kind.name(), s.offset, s.count]))
            .collect();
        meta["sections"] = Value::Array(secs);
        meta["version"] = json!(VERSION);
        let footer = serde_json::to_vec(&meta)?;
        self.sum.update(&footer);
        let len = footer.len() as u64;
        self.sum.update(&len.to_le_bytes());
        let sum = self.sum.finish();
        self.w.write_all(&footer)?;
        self.w.write_all(&len.to_le_bytes())?;
        self.w.write_all(&sum.to_le_bytes())?;
        self.w.write_all(MAGIC)?;
        self.w.flush()?;
        Ok((self.w, self.off + len + 24))
    }
}

/// Reads a snapshot file: its footer first, then the sections in file order.
pub struct SnapReader {
    f: File,
    meta: Value,
    sections: Vec<Section>,
    next: usize,
    sum: Checksum,
    footer: Vec<u8>,
    expect: u64,
    data_end: u64,
}

impl SnapReader {
    /// Opens a snapshot and reads its footer (the sections are read on demand).
    pub fn open(path: &Path) -> Result<Self> {
        let md = std::fs::symlink_metadata(path).with_context(|| format!("snapshot {}", path.display()))?;
        if !md.file_type().is_file() {
            bail!("snapshot {}: not a regular file", path.display());
        }
        let mut f = File::open(path).with_context(|| format!("snapshot {}", path.display()))?;
        let size = f.metadata()?.len();
        if size < 8 + 24 {
            bail!("snapshot {}: truncated", path.display());
        }
        let mut head = [0u8; 8];
        f.read_exact(&mut head)?;
        let mut tail = [0u8; 24];
        f.seek(SeekFrom::End(-24))?;
        f.read_exact(&mut tail)?;
        if &head != MAGIC || &tail[16..] != MAGIC {
            bail!("snapshot {}: not an oxinfer session snapshot (or another version)", path.display());
        }
        let flen = u64::from_le_bytes(tail[..8].try_into().unwrap());
        let expect = u64::from_le_bytes(tail[8..16].try_into().unwrap());
        if flen > MAX_FOOTER || flen + 24 + 8 > size {
            bail!("snapshot {}: bad footer", path.display());
        }
        let data_end = size - 24 - flen;
        f.seek(SeekFrom::Start(data_end))?;
        let mut footer = vec![0u8; flen as usize];
        f.read_exact(&mut footer)?;
        let meta: Value = serde_json::from_slice(&footer).with_context(|| format!("snapshot {}: footer", path.display()))?;
        if meta.get("version").and_then(Value::as_u64) != Some(VERSION) {
            bail!("snapshot {}: unsupported format version", path.display());
        }
        let mut sections = Vec::new();
        let mut off = 8u64;
        for s in meta
            .get("sections")
            .and_then(Value::as_array)
            .context("snapshot: no section table")?
        {
            let a = s.as_array().filter(|a| a.len() == 4).context("snapshot: bad section entry")?;
            let name = a[0].as_str().context("snapshot: section name")?.to_string();
            let kind = a[1].as_str().and_then(Kind::parse).context("snapshot: section type")?;
            let (offset, count) = (a[2].as_u64().context("offset")?, a[3].as_u64().context("count")?);
            let bytes = count.checked_mul(kind.size()).context("snapshot: section size")?;
            if offset != off || offset.checked_add(bytes).is_none_or(|e| e > data_end) {
                bail!("snapshot {}: section {name} out of place", path.display());
            }
            off += bytes;
            sections.push(Section { name, kind, offset, count });
        }
        if off != data_end {
            bail!("snapshot {}: sections do not cover the data", path.display());
        }
        f.seek(SeekFrom::Start(8))?;
        let mut sum = Checksum::new();
        sum.update(MAGIC);
        Ok(Self {
            f,
            meta,
            sections,
            next: 0,
            sum,
            footer,
            expect,
            data_end,
        })
    }

    /// The footer's metadata.
    pub fn meta(&self) -> &Value {
        &self.meta
    }

    /// Bytes of section data.
    pub fn data_bytes(&self) -> u64 {
        self.data_end - 8
    }

    /// Whether the next section is `name`.
    pub fn next_is(&self, name: &str) -> bool {
        self.sections.get(self.next).is_some_and(|s| s.name == name)
    }

    fn take<T: Copy + Default>(&mut self, name: &str, kind: Kind, count: Option<u64>, max: u64) -> Result<Vec<T>> {
        let s = self
            .sections
            .get(self.next)
            .with_context(|| format!("snapshot: missing section {name}"))?
            .clone();
        if s.name != name || s.kind != kind {
            bail!(
                "snapshot: expected section {name} ({}), found {} ({})",
                kind.name(),
                s.name,
                s.kind.name()
            );
        }
        match count {
            Some(c) if c != s.count => bail!("snapshot: section {name} has {} elements, the model expects {c}", s.count),
            None if s.count > max => bail!("snapshot: section {name} too large ({} elements)", s.count),
            _ => {}
        }
        self.next += 1;
        let mut v = vec![T::default(); s.count as usize];
        // SAFETY: plain numbers, every bit pattern valid.
        let raw = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v.as_slice())) };
        self.f.read_exact(raw).with_context(|| format!("snapshot: section {name}"))?;
        self.sum.update(raw);
        if cfg!(target_endian = "big") {
            let sz = std::mem::size_of::<T>();
            for c in raw.chunks_mut(sz) {
                c.reverse();
            }
        }
        Ok(v)
    }

    /// The next section, `name`, of exactly `count` f32 values.
    pub fn f32s(&mut self, name: &str, count: usize) -> Result<Vec<f32>> {
        self.take(name, Kind::F32, Some(count as u64), 0)
    }
    /// The next section into `out` (exactly its length).
    pub fn f32s_into(&mut self, name: &str, out: &mut [f32]) -> Result<()> {
        let v = self.f32s(name, out.len())?;
        out.copy_from_slice(&v);
        Ok(())
    }
    /// The next section, `name`, of at most `max` u32 values.
    pub fn u32s(&mut self, name: &str, max: usize) -> Result<Vec<u32>> {
        self.take(name, Kind::U32, None, max as u64)
    }
    /// The next section, `name`, of at most `max` u64 values.
    pub fn u64s(&mut self, name: &str, max: usize) -> Result<Vec<u64>> {
        self.take(name, Kind::U64, None, max as u64)
    }
    /// The next section, `name`, of exactly `count` u64 values.
    pub fn u64s_exact(&mut self, name: &str, count: usize) -> Result<Vec<u64>> {
        self.take(name, Kind::U64, Some(count as u64), 0)
    }

    /// Checks that every section was read and that the checksum matches.
    pub fn finish(mut self) -> Result<()> {
        if self.next != self.sections.len() {
            bail!("snapshot: {} sections were not read", self.sections.len() - self.next);
        }
        self.sum.update(&self.footer);
        self.sum.update(&(self.footer.len() as u64).to_le_bytes());
        if self.sum.finish() != self.expect {
            bail!("snapshot: checksum mismatch (corrupted file)");
        }
        Ok(())
    }
}

/// A model's decoding state that can be written to and read from a snapshot. `read_state` is
/// called on a fresh state (a new session of the same model) and must check every length.
pub trait SessionState {
    fn write_state<W: Write>(&self, w: &mut SnapWriter<W>) -> Result<()>;
    fn read_state(&mut self, r: &mut SnapReader) -> Result<()>;
    /// Whether the state can be cut back to any earlier position (no recurrent state).
    fn truncatable(&self) -> bool {
        true
    }
}

/// Writes a KV cache as the sections `name.k` and `name.v` (its `len` positions).
pub(crate) fn write_kv<W: Write>(w: &mut SnapWriter<W>, name: &str, c: &crate::model::LayerCache) -> Result<()> {
    let (k, v) = c.data();
    w.f32s(&format!("{name}.k"), &k[..c.len * c.n_kv * c.hd])?;
    w.f32s(&format!("{name}.v"), &v[..c.len * c.n_kv * c.hd])
}

/// Reads a KV cache written by [`write_kv`] into the fresh cache `c`, of `len` positions.
pub(crate) fn read_kv(r: &mut SnapReader, name: &str, c: &mut crate::model::LayerCache, len: usize) -> Result<()> {
    let n = len.checked_mul(c.n_kv * c.hd).context("snapshot: KV size")?;
    let k = r.f32s(&format!("{name}.k"), n)?;
    let v = r.f32s(&format!("{name}.v"), n)?;
    c.truncate(0);
    c.push(&k, &v, len);
    Ok(())
}

/// The default prompt-cache directory: `prompt-cache` beside the download cache
/// (`~/.cache/oxinfer/prompt-cache`, or the `$OXINFER_CACHE` directory's parent; on iOS in the
/// app's Library/Caches).
pub fn default_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("OXINFER_PROMPT_CACHE") {
        return PathBuf::from(p);
    }
    if let Some(p) = std::env::var_os("OXINFER_CACHE") {
        let p = PathBuf::from(p);
        return p.parent().map(Path::to_path_buf).unwrap_or(p).join("prompt-cache");
    }
    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| ".".into()));
    if cfg!(target_os = "ios") {
        return home.join("Library").join("Caches").join("oxinfer").join("prompt-cache");
    }
    home.join(".cache").join("oxinfer").join("prompt-cache")
}

/// Creates `dir` (and its parents) private to the user, or makes an existing one private.
pub fn private_dir(dir: &Path) -> Result<()> {
    if let Ok(md) = std::fs::symlink_metadata(dir) {
        if !md.file_type().is_dir() {
            bail!("{}: not a directory (or a symbolic link)", dir.display());
        }
    } else {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let md = std::fs::metadata(dir)?;
        if md.permissions().mode() & 0o777 != 0o700 {
            // Fails unless the directory is ours (e.g. /tmp, /dev/shm themselves).
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).with_context(|| {
                format!(
                    "{}: cannot make the directory private (mode 0700); use a directory of your own, not a shared one like /tmp",
                    dir.display()
                )
            })?;
        }
    }
    Ok(())
}

/// Writes a snapshot atomically: through `write` into a private temporary file in `path`'s
/// directory, renamed into place.
pub fn write_atomic(path: &Path, write: impl FnOnce(&mut std::io::BufWriter<File>) -> Result<()>) -> Result<u64> {
    let dir = path.parent().context("snapshot path has no directory")?;
    let tmp = dir.join(format!(
        ".{}.{}.tmp",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("snapshot"),
        std::process::id()
    ));
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    let _ = std::fs::remove_file(&tmp);
    let f = o.open(&tmp).with_context(|| format!("create {}", tmp.display()))?;
    let mut bw = std::io::BufWriter::with_capacity(4 << 20, f);
    let r = write(&mut bw).and_then(|_| {
        let f = bw.into_inner().map_err(|e| e.into_error())?;
        Ok(f.metadata()?.len())
    });
    match r {
        Ok(n) => {
            std::fs::rename(&tmp, path).with_context(|| format!("rename to {}", path.display()))?;
            Ok(n)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// One snapshot of a [`PromptCache`].
#[derive(Clone, Debug)]
pub struct CacheEntry {
    pub path: PathBuf,
    /// The model fingerprint hash it belongs to.
    pub model: String,
    pub tokens: Vec<u32>,
    pub bytes: u64,
    /// Last use (file modification time).
    pub used: std::time::SystemTime,
}

/// A directory of session snapshots keyed by model and token prefix, least recently used ones
/// removed beyond a size cap (see the module docs).
#[derive(Debug)]
pub struct PromptCache {
    dir: PathBuf,
    cap: u64,
    entries: Vec<CacheEntry>,
}

/// Default size cap of a prompt cache.
pub const DEFAULT_CAP_GB: f64 = 20.0;

impl PromptCache {
    /// Opens (creating it private if needed) the cache in `dir` with a cap of `cap` bytes, and
    /// indexes the snapshots there.
    pub fn open(dir: &Path, cap: u64) -> Result<Self> {
        private_dir(dir)?;
        let mut c = Self {
            dir: dir.to_path_buf(),
            cap,
            entries: Vec::new(),
        };
        c.rescan();
        Ok(c)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn entries(&self) -> &[CacheEntry] {
        &self.entries
    }

    /// Bytes of all snapshots.
    pub fn bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bytes).sum()
    }

    /// Re-reads the directory (snapshots written or removed by other processes).
    pub fn rescan(&mut self) {
        self.entries.clear();
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some(EXT) {
                continue;
            }
            if let Ok(entry) = Self::index(&p) {
                self.entries.push(entry);
            }
        }
    }

    fn index(p: &Path) -> Result<CacheEntry> {
        let md = std::fs::symlink_metadata(p)?;
        let mut r = SnapReader::open(p)?;
        let model = r.meta()["model"].as_str().context("no model")?.to_string();
        let n = r.meta()["tokens"].as_u64().context("no token count")? as usize;
        let tokens = r.u32s("tokens", n)?;
        if tokens.len() != n {
            bail!("token count mismatch");
        }
        Ok(CacheEntry {
            path: p.to_path_buf(),
            model,
            tokens,
            bytes: md.len(),
            used: md.modified().unwrap_or(std::time::UNIX_EPOCH),
        })
    }

    /// The file name of the snapshot of `tokens` for `model`.
    pub fn path_for(&self, model: &str, tokens: &[u32]) -> PathBuf {
        let h = hash64(&[model.as_bytes(), &tokens_bytes(tokens)]);
        self.dir.join(format!("{h:016x}-{}.{EXT}", tokens.len()))
    }

    /// The snapshot of `model` that saves the most of `ids`: `(entry, tokens usable)`. A
    /// snapshot is usable up to its common prefix with `ids` when the state can be cut back
    /// (`truncatable`), else only when `ids` starts with all of it; at least one token of `ids`
    /// is left to feed.
    pub fn best(&self, model: &str, ids: &[u32], truncatable: bool) -> Option<(&CacheEntry, usize)> {
        let mut best: Option<(&CacheEntry, usize)> = None;
        for e in self.entries.iter().filter(|e| e.model == model) {
            let common = e.tokens.iter().zip(ids).take_while(|(a, b)| a == b).count();
            let usable = if truncatable {
                common.min(ids.len().saturating_sub(1))
            } else if common == e.tokens.len() && common < ids.len() {
                common
            } else {
                0
            };
            if usable > 0 && best.is_none_or(|b| usable > b.1 || (usable == b.1 && e.tokens.len() < b.0.tokens.len())) {
                best = Some((e, usable));
            }
        }
        best
    }

    /// Whether a snapshot of exactly `tokens` (or, `truncatable`, of tokens starting with them)
    /// is stored for `model`.
    pub fn covers(&self, model: &str, tokens: &[u32], truncatable: bool) -> bool {
        self.entries
            .iter()
            .any(|e| e.model == model && (e.tokens == tokens || (truncatable && e.tokens.starts_with(tokens))))
    }

    /// Marks a snapshot as just used.
    pub fn touch(&mut self, path: &Path) {
        let now = std::time::SystemTime::now();
        if let Ok(f) = std::fs::OpenOptions::new().write(true).open(path) {
            let _ = f.set_modified(now);
        }
        if let Some(e) = self.entries.iter_mut().find(|e| e.path == path) {
            e.used = now;
        }
    }

    /// Stores a snapshot of `tokens` for `model`, written by `write`; with `truncatable`,
    /// snapshots of this model whose tokens are a prefix of `tokens` are removed (this one
    /// serves them). Then the least recently used snapshots go until the cache fits its cap.
    pub fn store(
        &mut self,
        model: &str,
        tokens: &[u32],
        truncatable: bool,
        write: impl FnOnce(&mut std::io::BufWriter<File>) -> Result<()>,
    ) -> Result<PathBuf> {
        let path = self.path_for(model, tokens);
        let bytes = write_atomic(&path, write)?;
        self.entries.retain(|e| e.path != path);
        if truncatable {
            let (keep, drop): (Vec<_>, Vec<_>) = std::mem::take(&mut self.entries)
                .into_iter()
                .partition(|e| !(e.model == model && tokens.starts_with(&e.tokens)));
            for e in drop {
                let _ = std::fs::remove_file(&e.path);
            }
            self.entries = keep;
        }
        self.entries.push(CacheEntry {
            path: path.clone(),
            model: model.to_string(),
            tokens: tokens.to_vec(),
            bytes,
            used: std::time::SystemTime::now(),
        });
        self.evict(Some(&path));
        Ok(path)
    }

    /// Removes the least recently used snapshots (never `keep`) until the cache fits its cap.
    pub fn evict(&mut self, keep: Option<&Path>) {
        while self.bytes() > self.cap {
            let Some(i) = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| Some(e.path.as_path()) != keep)
                .min_by_key(|(_, e)| e.used)
                .map(|(i, _)| i)
            else {
                break;
            };
            let e = self.entries.remove(i);
            let _ = std::fs::remove_file(&e.path);
        }
    }

    /// Forgets a snapshot that could not be read (and removes its file).
    pub fn remove(&mut self, path: &Path) {
        self.entries.retain(|e| e.path != path);
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("oxinfer-snap-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn round_trip_and_corruption() {
        let dir = tmp("rt");
        private_dir(&dir).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
        }
        let p = dir.join("a.oxsnap");
        let a: Vec<f32> = (0..1001).map(|i| i as f32 * 0.5 - 3.0).collect();
        write_atomic(&p, |f| {
            let mut w = SnapWriter::new(f)?;
            w.u32s("tokens", &[1, 2, 3])?;
            w.f32s("a", &a)?;
            w.u64s("n", &[7, u64::MAX])?;
            w.finish(json!({"model": "m", "tokens": 3}))?;
            Ok(())
        })
        .unwrap();
        let mut r = SnapReader::open(&p).unwrap();
        assert_eq!(r.meta()["model"], "m");
        assert_eq!(r.u32s("tokens", 10).unwrap(), [1, 2, 3]);
        assert!(r.f32s("a", 1000).is_err(), "length checked before reading");
        let mut r = SnapReader::open(&p).unwrap();
        r.u32s("tokens", 10).unwrap();
        assert_eq!(r.f32s("a", 1001).unwrap(), a);
        assert_eq!(r.u64s("n", 2).unwrap(), [7, u64::MAX]);
        r.finish().unwrap();
        // A flipped bit in the data fails the checksum.
        let mut b = std::fs::read(&p).unwrap();
        b[100] ^= 4;
        std::fs::write(&p, &b).unwrap();
        let mut r = SnapReader::open(&p).unwrap();
        r.u32s("tokens", 10).unwrap();
        r.f32s("a", 1001).unwrap();
        r.u64s("n", 2).unwrap();
        assert!(r.finish().is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cache_prefixes_and_eviction() {
        let dir = tmp("cache");
        let write = |toks: &[u32], pad: usize| {
            let toks = toks.to_vec();
            move |f: &mut std::io::BufWriter<File>| -> Result<()> {
                let mut w = SnapWriter::new(f)?;
                w.u32s("tokens", &toks)?;
                w.f32s("pad", &vec![0.0; pad])?;
                w.finish(json!({"model": "m", "tokens": toks.len()}))?;
                Ok(())
            }
        };
        let mut c = PromptCache::open(&dir, 1 << 20).unwrap();
        c.store("m", &[1, 2, 3, 4], false, write(&[1, 2, 3, 4], 10)).unwrap();
        c.store("m", &[1, 2, 3, 4, 5, 6], false, write(&[1, 2, 3, 4, 5, 6], 10)).unwrap();
        // Recurrent: only whole snapshots the prompt starts with.
        let (e, n) = c.best("m", &[1, 2, 3, 4, 5, 6, 7], false).unwrap();
        assert_eq!((e.tokens.len(), n), (6, 6));
        let (e, n) = c.best("m", &[1, 2, 3, 4, 5, 9], false).unwrap();
        assert_eq!((e.tokens.len(), n), (4, 4));
        assert!(c.best("m", &[1, 2, 3, 4], false).is_none(), "a token must be left to feed");
        assert!(c.best("other", &[1, 2, 3, 4, 5], false).is_none());
        // Truncatable: any common prefix.
        let (_, n) = c.best("m", &[1, 2, 3, 4, 5, 9], true).unwrap();
        assert_eq!(n, 5);
        // A truncatable store replaces its prefixes; reopening finds the snapshots again.
        c.store("m", &[1, 2, 3, 4, 5, 6, 7], true, write(&[1, 2, 3, 4, 5, 6, 7], 10))
            .unwrap();
        assert_eq!(c.entries().len(), 1);
        let c2 = PromptCache::open(&dir, 1 << 20).unwrap();
        assert_eq!(c2.entries().len(), 1);
        assert_eq!(c2.entries()[0].tokens, [1, 2, 3, 4, 5, 6, 7]);
        // The cap evicts the least recently used.
        let mut c = PromptCache::open(&dir, 300_000).unwrap();
        c.store("m", &[9, 9], false, write(&[9, 9], 50_000)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        c.store("m", &[8, 8], false, write(&[8, 8], 50_000)).unwrap();
        assert!(c.entries().len() <= 2);
        assert!(c.entries().iter().any(|e| e.tokens == [8, 8]));
        std::fs::remove_dir_all(&dir).ok();
    }
}

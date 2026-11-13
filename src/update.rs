//! Self-update from the project's GitHub releases (`oxinfer --check-update`, `oxinfer --update`).
//!
//! The latest release is read from the GitHub API, its archive for this platform downloaded
//! (the names `release-binaries.yml` gives them), unpacked in memory (gzip with `compcol`, the
//! tar or zip container with `fstool`), and the running executable replaced by the new one.
//!
//! What vouches for the download is GitHub over TLS and the size the API lists: the releases
//! are not signed, and the asset's SHA-256 digest is not checked.

use anyhow::{Context, Result, anyhow, bail};
use std::path::{Path, PathBuf};

/// The repository whose releases carry the binaries.
pub const REPO: &str = "KarpelesLab/oxinfer";

/// The version this binary was built from.
pub const CURRENT: &str = env!("CARGO_PKG_VERSION");

/// The suffix of this platform's release archive (`oxinfer-<tag>-<suffix>.tar.gz` / `.zip`), or
/// None where no binary is published.
pub fn platform_suffix() -> Option<&'static str> {
    if cfg!(target_os = "fullrust") {
        Some("linux-x86_64-static")
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu")) {
        Some("linux-x86_64")
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64", target_env = "gnu")) {
        Some("linux-aarch64")
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Some("windows-x86_64")
    } else if cfg!(all(target_os = "windows", target_arch = "aarch64")) {
        Some("windows-aarch64")
    } else if cfg!(target_os = "macos") {
        Some("macos-universal")
    } else {
        None
    }
}

/// One file attached to a release.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct Asset {
    pub name: String,
    pub browser_download_url: String,
    pub size: u64,
    /// `sha256:<hex>`, when GitHub reports it.
    #[serde(default)]
    pub digest: Option<String>,
}

/// A release as the GitHub API describes it.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct Release {
    pub tag_name: String,
    #[serde(default)]
    pub html_url: String,
    #[serde(default)]
    pub assets: Vec<Asset>,
}

impl Release {
    /// The release's version without its `v`.
    pub fn version(&self) -> &str {
        self.tag_name.strip_prefix('v').unwrap_or(&self.tag_name)
    }
    /// Whether it is newer than the running binary.
    pub fn is_newer(&self) -> bool {
        newer(self.version(), CURRENT)
    }
    /// This platform's archive.
    pub fn asset(&self) -> Option<&Asset> {
        let suffix = platform_suffix()?;
        let stem = format!("oxinfer-{}-{suffix}", self.tag_name);
        self.assets
            .iter()
            .find(|a| a.name == format!("{stem}.tar.gz") || a.name == format!("{stem}.zip"))
    }
}

/// The numeric fields of a version (`0.0.9`, `1.2.0-rc1` -> `[1, 2, 0]`).
fn fields(v: &str) -> Vec<u64> {
    v.split(['-', '+'])
        .next()
        .unwrap_or("")
        .split('.')
        .map(|f| f.parse().unwrap_or(0))
        .collect()
}

/// Whether version `a` is newer than `b` (numeric fields compared in order; a pre-release
/// suffix is ignored).
pub fn newer(a: &str, b: &str) -> bool {
    let (a, b) = (fields(a), fields(b));
    for i in 0..a.len().max(b.len()) {
        let (x, y) = (a.get(i).copied().unwrap_or(0), b.get(i).copied().unwrap_or(0));
        if x != y {
            return x > y;
        }
    }
    false
}

/// The latest published release (`GET /repos/<repo>/releases/latest`, unauthenticated: GitHub
/// allows 60 such requests an hour per address).
pub fn latest() -> Result<Release> {
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let body = crate::hf::http_get(&url).context("ask GitHub for the latest release")?;
    serde_json::from_slice(&body).with_context(|| format!("parse {url}"))
}

/// The `oxinfer` executable inside a release archive (`name`: the asset's file name).
pub fn unpack(name: &str, archive: Vec<u8>) -> Result<Vec<u8>> {
    let (container, member) = if name.ends_with(".zip") {
        (archive, "/oxinfer.exe")
    } else if name.ends_with(".tar.gz") {
        // A binary of a few hundred MB at most: a cap against a hostile archive.
        let tar =
            compcol::vec::decompress_to_vec_capped::<compcol::gzip::Gzip>(&archive, 2 << 30).map_err(|e| anyhow!("{name}: gzip: {e:?}"))?;
        (tar, "/oxinfer")
    } else {
        bail!("{name}: neither a .tar.gz nor a .zip");
    };
    let mut image = fstool::memconv::MemImage::open(container).map_err(|e| anyhow!("{name}: {e}"))?;
    let bin = image
        .read_file(member)
        .map_err(|e| anyhow!("{name}: no {member} in the archive ({e})"))?;
    if bin.is_empty() {
        bail!("{name}: {member} is empty");
    }
    Ok(bin)
}

/// Puts `bin` in the place of the executable at `exe`: written beside it, then swapped in by
/// renames, so an interrupted update leaves the old binary. The previous one is removed where
/// the OS allows it (not on Windows while it runs: it stays as `<name>.old`).
pub fn replace_executable(exe: &Path, bin: &[u8]) -> Result<()> {
    let dir = exe.parent().context("the executable has no directory")?;
    let name = exe.file_name().context("the executable has no name")?.to_string_lossy().to_string();
    let (new, old) = (dir.join(format!(".{name}.new")), dir.join(format!("{name}.old")));
    std::fs::write(&new, bin).with_context(|| format!("write {} (is the directory writable?)", new.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&new, std::fs::Permissions::from_mode(0o755))?;
    }
    // fullrust (libc-free Linux, not in the `unix` family) has no `PermissionsExt`: the mode
    // can only be copied, from the binary being replaced.
    #[cfg(target_os = "fullrust")]
    std::fs::set_permissions(&new, std::fs::metadata(exe)?.permissions())?;
    let _ = std::fs::remove_file(&old);
    let swap = std::fs::rename(exe, &old).and_then(|()| {
        std::fs::rename(&new, exe).inspect_err(|_| {
            // Put the old one back.
            let _ = std::fs::rename(&old, exe);
        })
    });
    if let Err(e) = swap {
        let _ = std::fs::remove_file(&new);
        return Err(e).with_context(|| format!("replace {}", exe.display()));
    }
    let _ = std::fs::remove_file(&old);
    Ok(())
}

/// What [`update`] did.
#[derive(Clone, Debug)]
pub enum Outcome {
    /// Already at the latest release (its version).
    UpToDate(String),
    /// Replaced the executable at the path with the version.
    Updated(PathBuf, String),
}

/// Downloads the latest release when it is newer than this binary (or `force`) and replaces the
/// running executable with it. `log` receives progress lines.
pub fn update(force: bool, log: &mut dyn FnMut(&str)) -> Result<Outcome> {
    let rel = latest()?;
    if !force && !rel.is_newer() {
        return Ok(Outcome::UpToDate(rel.version().to_string()));
    }
    let suffix = platform_suffix().context("no binary is published for this platform: build from source")?;
    let asset = rel
        .asset()
        .with_context(|| format!("release {} has no {suffix} archive ({})", rel.tag_name, rel.html_url))?;
    log(&format!("downloading {} ({:.1} MB)", asset.name, asset.size as f64 / 1e6));
    let archive = crate::hf::http_get(&asset.browser_download_url)?;
    if archive.len() as u64 != asset.size {
        bail!("{}: got {} bytes, the release lists {}", asset.name, archive.len(), asset.size);
    }
    let bin = unpack(&asset.name, archive)?;
    let exe = std::env::current_exe()
        .and_then(std::fs::canonicalize)
        .context("find the running executable")?;
    replace_executable(&exe, &bin)?;
    Ok(Outcome::Updated(exe, rel.version().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions() {
        assert!(newer("0.0.9", "0.0.8"));
        assert!(newer("0.1.0", "0.0.12"));
        assert!(newer("1.0", "0.9.9"));
        assert!(!newer("0.0.8", "0.0.8"));
        assert!(!newer("0.0.8", "0.0.9"));
        assert!(!newer("0.0.9-rc1", "0.0.9"));
        let rel: Release = serde_json::from_str(
            r#"{"tag_name":"v9.0.0","html_url":"u","assets":[
                {"name":"oxinfer-v9.0.0-linux-x86_64.tar.gz","browser_download_url":"a","size":3,"digest":"sha256:00"},
                {"name":"oxinfer-v9.0.0-linux-x86_64-static.tar.gz","browser_download_url":"s","size":3},
                {"name":"oxinfer-v9.0.0-linux-aarch64.tar.gz","browser_download_url":"b","size":3},
                {"name":"oxinfer-v9.0.0-macos-universal.tar.gz","browser_download_url":"c","size":3},
                {"name":"oxinfer-v9.0.0-windows-x86_64.zip","browser_download_url":"d","size":3},
                {"name":"oxinfer-v9.0.0-windows-aarch64.zip","browser_download_url":"e","size":3}]}"#,
        )
        .unwrap();
        assert_eq!(rel.version(), "9.0.0");
        assert!(rel.is_newer());
        if let Some(s) = platform_suffix() {
            let a = rel.asset().unwrap();
            assert!(a.name.starts_with(&format!("oxinfer-v9.0.0-{s}.")), "{}", a.name);
        }
    }

    /// A ustar archive holding one file.
    fn tar(name: &str, data: &[u8]) -> Vec<u8> {
        let mut h = [0u8; 512];
        h[..name.len()].copy_from_slice(name.as_bytes());
        h[100..108].copy_from_slice(b"0000755\0");
        h[108..116].copy_from_slice(b"0000000\0");
        h[116..124].copy_from_slice(b"0000000\0");
        h[124..136].copy_from_slice(format!("{:011o}\0", data.len()).as_bytes());
        h[136..148].copy_from_slice(b"00000000000\0");
        h[148..156].copy_from_slice(b"        ");
        h[156] = b'0';
        h[257..263].copy_from_slice(b"ustar\0");
        h[263..265].copy_from_slice(b"00");
        let sum: u32 = h.iter().map(|&b| b as u32).sum();
        h[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
        let mut out = h.to_vec();
        out.extend_from_slice(data);
        out.resize(out.len().div_ceil(512) * 512 + 1024, 0);
        out
    }

    #[test]
    fn unpacks_a_tar_gz_and_replaces_a_file() {
        let payload: Vec<u8> = (0..70_000u32).map(|i| (i * 31 % 251) as u8).collect();
        let gz = compcol::vec::compress_to_vec::<compcol::gzip::Gzip>(&tar("./oxinfer", &payload)).unwrap();
        assert_eq!(unpack("oxinfer-v1-linux-x86_64.tar.gz", gz).unwrap(), payload);
        assert!(unpack("oxinfer.rar", vec![0; 8]).is_err());
        let other = compcol::vec::compress_to_vec::<compcol::gzip::Gzip>(&tar("./README.md", b"x")).unwrap();
        assert!(unpack("oxinfer-v1-linux-x86_64.tar.gz", other).is_err());

        let dir = std::env::temp_dir().join(format!("oxinfer-update-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let exe = dir.join("oxinfer");
        std::fs::write(&exe, b"old").unwrap();
        replace_executable(&exe, &payload).unwrap();
        assert_eq!(std::fs::read(&exe).unwrap(), payload);
        let left: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(left.len(), 1, "{left:?}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

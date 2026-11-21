//! `oxinfer models list` / `oxinfer models rm`: the model cache ([`oxinfer::model_cache`]).

use anyhow::{Context, Result, bail};
use std::io::{BufRead, IsTerminal, Write};
use std::path::PathBuf;

pub const USAGE: &str = "usage: oxinfer models list [--json] [--cache-dir DIR]\n\
\x20      oxinfer models rm NAME... [--yes] [--cache-dir DIR]\n\
\n\
list: the folders of the model cache ($OXINFER_CACHE, default ~/.cache/oxinfer/hub) with their size,\n\
when oxinfer last loaded them, the specs that load them, and the disk's free space.\n\
rm: deletes cache folders, each named by its folder (org--name) or repo id (org/name); asks first\n\
unless --yes. Only folders directly inside the cache are removed.\n\
(`oxinfer --models` lists the tested models and what this machine can run.)";

fn gb(b: u64) -> String {
    if b >= 1_000_000_000 {
        format!("{:.1} GB", b as f64 / 1e9)
    } else {
        format!("{:.0} MB", b as f64 / 1e6)
    }
}

fn ago(t: Option<u64>) -> String {
    let Some(t) = t else { return "-".into() };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(t);
    let d = now.saturating_sub(t);
    match d {
        0..60 => "just now".into(),
        60..3600 => format!("{}m ago", d / 60),
        3600..86400 => format!("{}h ago", d / 3600),
        _ => format!("{}d ago", d / 86400),
    }
}

/// Runs the arguments after `models`.
pub fn run(args: &[String]) -> Result<()> {
    let mut cache_dir: Option<PathBuf> = None;
    let (mut json, mut yes) = (false, false);
    let mut rest = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--cache-dir" => cache_dir = Some(it.next().context("--cache-dir needs a value")?.into()),
            s if s.starts_with("--cache-dir=") => cache_dir = Some(s["--cache-dir=".len()..].into()),
            "--json" => json = true,
            "--yes" | "-y" => yes = true,
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return Ok(());
            }
            s if s.starts_with('-') => bail!("unknown option {s}\n{USAGE}"),
            s => rest.push(s.to_string()),
        }
    }
    let cache = cache_dir.as_deref();
    match rest.first().map(String::as_str) {
        Some("list" | "ls") => {
            if json {
                println!("{}", serde_json::to_string_pretty(&oxinfer::model_cache::to_json(cache)?)?);
                return Ok(());
            }
            let entries = oxinfer::model_cache::list(cache)?;
            let root = oxinfer::model_cache::root(cache);
            if entries.is_empty() {
                println!("No models in {} yet.", root.display());
            } else {
                let w = entries.iter().map(|e| e.name.len()).max().unwrap_or(4).max(4);
                println!("{:w$}  {:>8}  {:>10}  LOADS AS", "NAME", "SIZE", "LAST USED");
                for e in &entries {
                    let specs = e.specs().join(", ");
                    let specs = if e.usable {
                        specs
                    } else {
                        "(incomplete: no configuration or GGUF file)".into()
                    };
                    println!("{:w$}  {:>8}  {:>10}  {specs}", e.name, gb(e.bytes), ago(e.last_used));
                }
            }
            let total: u64 = entries.iter().map(|e| e.bytes).sum();
            match oxinfer::model_cache::disk_space(cache) {
                Some((free, size)) => println!(
                    "\n{} in {} folders in {}; {} free of {} on its disk.",
                    gb(total),
                    entries.len(),
                    root.display(),
                    gb(free),
                    gb(size)
                ),
                None => println!("\n{} in {} folders in {}.", gb(total), entries.len(), root.display()),
            }
            Ok(())
        }
        Some("rm" | "remove" | "delete") => {
            let names = &rest[1..];
            if names.is_empty() {
                bail!("models rm: name the folders or repo ids to delete\n{USAGE}");
            }
            // Everything is checked before anything is deleted.
            let mut found = Vec::new();
            for n in names {
                let p = oxinfer::model_cache::locate(n, cache)?;
                let e = oxinfer::model_cache::entry_at(&p)?;
                found.push((n.clone(), e));
            }
            for (_, e) in &found {
                println!("{}  {}", gb(e.bytes), e.path.display());
            }
            if !yes {
                if !std::io::stdin().is_terminal() {
                    bail!("not deleting without confirmation: pass --yes");
                }
                print!("Delete {} folder{}? [y/N] ", found.len(), if found.len() == 1 { "" } else { "s" });
                std::io::stdout().flush()?;
                let mut line = String::new();
                std::io::stdin().lock().read_line(&mut line)?;
                if !matches!(line.trim(), "y" | "Y" | "yes") {
                    println!("Nothing deleted.");
                    return Ok(());
                }
            }
            let mut freed = 0;
            for (n, _) in &found {
                freed += oxinfer::model_cache::remove(n, cache)?;
                println!("deleted {n}");
            }
            println!("{} freed.", gb(freed));
            Ok(())
        }
        _ => {
            eprintln!("{USAGE}");
            bail!("models: expected list or rm");
        }
    }
}

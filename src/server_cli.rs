//! The option parsing of `oxinfer serve` / `oxinfer web` (one server, see `web_cli`).

use anyhow::{Context, Result, bail};
use oxinfer::server::ServerOptions;

/// A caller's own flags: `(flag, value getter)`, true when it handled the flag.
pub type ExtraFlags<'a> = dyn FnMut(&str, &mut dyn FnMut() -> Result<String>) -> Result<bool> + 'a;

/// Parses the options shared by `serve` and `web` into `o` (after its defaults);
/// `extra` takes the flags of the caller first (true: handled).
pub fn parse_options(o: &mut ServerOptions, args: &[String], usage: &str, command: &str, extra: &mut ExtraFlags) -> Result<()> {
    // The command line's server takes the models requests name (the library's default is off).
    o.set("auto_models", "1")?;
    if let Ok(k) = std::env::var("OXINFER_API_KEY")
        && !k.is_empty()
    {
        o.set("api_key", &k)?;
    }
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let flag = a.strip_prefix("--").or_else(|| a.strip_prefix('-')).unwrap_or(a);
        let (flag, inline) = match flag.split_once('=') {
            Some((f, v)) => (f, Some(v.to_string())),
            None => (flag, None),
        };
        let mut val = || -> Result<String> {
            match &inline {
                Some(v) => Ok(v.clone()),
                None => it.next().cloned().with_context(|| format!("{a} needs a value")),
            }
        };
        if extra(flag, &mut val)? {
            continue;
        }
        match flag {
            "h" | "help" => {
                eprintln!("{usage}");
                std::process::exit(2);
            }
            "m" | "model" => o.set("model", &val()?)?,
            "l" => o.set("listen", &val()?)?,
            "preload" | "download" | "self-signed" | "local-media" => o.set(&flag.replace('-', "_"), "1")?,
            "no-remote-media" => o.set("remote_media", "0")?,
            "no-update-check" => o.set("update_check", "0")?,
            "no-auto-models" => o.set("auto_models", "0")?,
            "auto-models" => o.set("auto_models", "1")?,
            "idle-unload" => o.set("idle_unload_min", &val()?)?,
            "quiet" | "q" => o.set("log", "0")?,
            f if f.starts_with(|c: char| c.is_ascii_alphabetic()) => {
                let key = f.replace('-', "_");
                if !ServerOptions::KEYS.contains(&key.as_str()) || key == "web" {
                    bail!("unknown option {a} (oxinfer {command} --help)");
                }
                o.set(&key, &val()?)?
            }
            _ => bail!("unexpected argument {a} (oxinfer {command} --help)"),
        }
    }
    Ok(())
}

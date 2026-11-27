//! `oxinfer split-worker`: the last layers of a model, served to the front that runs the first
//! ones ([`oxinfer::split`], docs/cluster.md).

use anyhow::{Context, Result, bail};
use oxinfer::split::{Worker, WorkerOptions};

pub const USAGE: &str = "usage: oxinfer split-worker MODEL --listen ADDR:PORT --key KEY [options]\n\
\n\
Runs the last layers of MODEL (a path or a repo id; a dense Qwen, Qwen3.5 / 3.8 or Llama model)\n\
for a front on another machine, which runs the first ones: `oxinfer chat MODEL --split\n\
ADDR:PORT --split-key KEY`, or a served model with the options split=ADDR:PORT,split_key=KEY.\n\
Both machines hold the same model files. See docs/cluster.md.\n\
\n\
options:\n\
\x20 --listen ADDR:PORT   the address to listen on (default 127.0.0.1:8090): the direct link's\n\
\x20 --key KEY            the key the front must give (default $OXINFER_SPLIT_KEY); required\n\
\x20 --from K             the first layer run here (default: half the layers); the front loads\n\
\x20                      the layers before it\n\
\x20 --quant Q, --device D, --cache-dir DIR, --threads N   as for the other commands\n\
\x20 --quiet              no line per connection";

pub fn run(args: &[String]) -> Result<()> {
    let mut o = WorkerOptions::default();
    if let Ok(k) = std::env::var("OXINFER_SPLIT_KEY") {
        o.key = k;
    }
    let mut model = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let flag = a.strip_prefix("--");
        let (flag, inline) = match flag.and_then(|f| f.split_once('=')) {
            Some((f, v)) => (Some(f), Some(v.to_string())),
            None => (flag, None),
        };
        let mut val = || -> Result<String> {
            match &inline {
                Some(v) => Ok(v.clone()),
                None => it.next().cloned().with_context(|| format!("{a} needs a value")),
            }
        };
        match flag {
            Some("help") => {
                eprintln!("{USAGE}");
                std::process::exit(2);
            }
            Some("listen") => o.listen = val()?,
            Some("key") => o.key = val()?,
            Some("from") => o.from = Some(val()?.parse().context("--from: expected a layer index")?),
            Some("quiet") => o.log = false,
            Some(f @ ("quant" | "device" | "cache-dir" | "threads")) => o.engine.set(&f.replace('-', "_"), &val()?)?,
            Some(_) => bail!("unknown option {a} (oxinfer split-worker --help)"),
            None if a == "-h" => {
                eprintln!("{USAGE}");
                std::process::exit(2);
            }
            None if model.is_none() => model = Some(a.clone()),
            None => bail!("unexpected argument {a} (oxinfer split-worker --help)"),
        }
    }
    let Some(model) = model else {
        bail!("{USAGE}");
    };
    if let Some(t) = o.engine.threads {
        oxinfer::pool::init_with(oxinfer::pool::PoolOptions::threads(t));
    }
    let t0 = std::time::Instant::now();
    let w = Worker::start(&model, &o)?;
    let (k, n) = w.layers();
    eprintln!(
        "[split-worker: layers {k}..{n} of {model} on {} ({:.1}s); the front takes layers 0..{k}]",
        w.addr(),
        t0.elapsed().as_secs_f64()
    );
    w.wait();
    Ok(())
}

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use vitter::Vitter;

#[derive(Parser)]
struct Opts {
    #[clap(short, long)]
    input: Vec<PathBuf>,
    #[clap(short, long)]
    output_dir: Option<PathBuf>,
}

fn main() -> Result<()> {
    env_logger::init();
    let opts = Opts::parse();
    let vitter = Vitter::default()?;
    let output_dir = opts.output_dir.unwrap_or_default();
    vitter.image_to_text(&opts.input, &output_dir)?;
    Ok(())
}

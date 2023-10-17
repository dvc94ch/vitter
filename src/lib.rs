use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::io::Reader as ImageReader;
use image::{DynamicImage, RgbImage, RgbaImage};
use ndarray::{Array, ArrayBase, Dim, IxDynImpl, OwnedRepr};
use ort::{Environment, ExecutionProvider, Session, SessionBuilder};
use resvg::usvg::{TreeParsing, TreeTextToPath};
use resvg::{tiny_skia, usvg};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const ENCODER: &[u8] = include_bytes!("../models/encoder_model_quantized.onnx");
const DECODER: &[u8] = include_bytes!("../models/decoder_model_merged_quantized.onnx");
const TOKENIZER: &[u8] = include_bytes!("../models/tokenizer.json");

pub struct Vitter {
    encoder: Session,
    decoder: Session,
    tokenizer: Tokenizer,
    batch_size: usize,
    img_size: u32,
    img_rescale: f32,
    img_mean: f32,
    img_std: f32,
}

type Tensor = ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>;

impl Vitter {
    pub fn new(encoder: &[u8], decoder: &[u8], tokenizer: &[u8]) -> Result<Self> {
        let environment = Environment::builder()
            .with_name("vitter")
            .with_execution_providers([ExecutionProvider::CPU(Default::default())])
            .build()?
            .into_arc();
        let encoder = SessionBuilder::new(&environment)?.with_model_from_memory(encoder)?;
        let decoder = SessionBuilder::new(&environment)?.with_model_from_memory(decoder)?;
        let tokenizer = Tokenizer::from_bytes(tokenizer).unwrap();
        Ok(Self {
            encoder,
            decoder,
            tokenizer,
            batch_size: 10,
            img_size: 224,
            img_rescale: 0.00392156862745098,
            img_mean: 0.5,
            img_std: 0.5,
        })
    }

    pub fn from_path(encoder: &Path, decoder: &Path, tokenizer: &Path) -> Result<Self> {
        let encoder = std::fs::read(encoder)?;
        let decoder = std::fs::read(decoder)?;
        let tokenizer = std::fs::read(tokenizer)?;
        Self::new(&encoder, &decoder, &tokenizer)
    }

    pub fn default() -> Result<Self> {
        Self::new(ENCODER, DECODER, TOKENIZER)
    }

    pub fn read_svg(&self, path: &Path) -> Result<RgbImage> {
        let rtree = {
            let opt = usvg::Options::default();
            let mut fontdb = fontdb::Database::new();
            fontdb.load_system_fonts();

            let svg_data = std::fs::read(path)?;
            let mut tree = usvg::Tree::from_data(&svg_data, &opt)
                .map_err(|err| anyhow::anyhow!("invalid svg: {err}"))?;
            tree.convert_text(&fontdb);
            resvg::Tree::from_usvg(&tree)
        };

        let pixmap_size = rtree.size.to_int_size();
        let mut pixmap = tiny_skia::Pixmap::new(pixmap_size.width(), pixmap_size.height()).unwrap();
        rtree.render(tiny_skia::Transform::default(), &mut pixmap.as_mut());
        let img =
            RgbaImage::from_raw(pixmap_size.width(), pixmap_size.height(), pixmap.take()).unwrap();
        Ok(DynamicImage::from(img)
            .resize_to_fill(self.img_size, self.img_size, FilterType::Triangle)
            .into_rgb8())
    }

    pub fn read_image(&self, path: &Path) -> Result<RgbImage> {
        if path.extension().map(|e| e.to_str()).flatten() == Some("svg") {
            return self.read_svg(path);
        }
        Ok(ImageReader::open(path)?
            .decode()?
            .resize_to_fill(self.img_size, self.img_size, FilterType::Triangle)
            .into_rgb8())
    }

    pub fn prepare(&self, batch: &[RgbImage]) -> Tensor {
        let mut input =
            Array::zeros((batch.len(), 3, self.img_size as _, self.img_size as _)).into_dyn();
        for (b, img) in batch.iter().enumerate() {
            for c in 0..3 {
                for y in 0..self.img_size {
                    for x in 0..self.img_size {
                        let value = if x < img.width() && y < img.height() {
                            let pixel = img.get_pixel(x, y).0[c];
                            (pixel as f32 * self.img_rescale - self.img_mean) / self.img_std
                        } else {
                            0.0
                        };
                        input[[b, c, y as usize, x as usize]] = value;
                    }
                }
            }
        }
        input
    }

    pub fn encode(&self, input: &Tensor) -> Result<Tensor> {
        let input_values = &input.as_standard_layout();
        let outputs = self
            .encoder
            .run(ort::inputs!["pixel_values" => input_values])?;
        let tensor = outputs["last_hidden_state"].extract_tensor::<f32>()?;
        let a = tensor.view().slice(ndarray::s![.., 0, 0]).len();
        let b = tensor.view().slice(ndarray::s![0, .., 0]).len();
        let c = tensor.view().slice(ndarray::s![0, 0, ..]).len();
        let mut output = Array::zeros((a, b, c)).into_dyn();
        for x in 0..a {
            for y in 0..b {
                for z in 0..c {
                    output[[x, y, z]] = tensor.view()[[x, y, z]];
                }
            }
        }
        Ok(output)
    }

    pub fn decode_one(
        &self,
        encoder_hidden_states: &Tensor,
        tokens: &mut [Vec<u32>],
    ) -> Result<bool> {
        // TODO: use cache branch
        // TODO: temp, top_k top_p, beam search
        let mut input_ids = Array::zeros((tokens.len(), tokens[0].len())).into_dyn();
        for (i, batch) in tokens.iter().enumerate() {
            for (j, token) in batch.iter().enumerate() {
                input_ids[[i, j]] = *token as i64;
            }
        }
        let mut use_cache_branch = Array::default(1).into_dyn();
        use_cache_branch[0] = false;
        let input_ids = &input_ids.as_standard_layout();
        let encoder_hidden_states = &encoder_hidden_states.as_standard_layout();
        let use_cache_branch = &use_cache_branch.as_standard_layout();
        let outputs = self.decoder.run(ort::inputs![
            "input_ids" => input_ids,
            "encoder_hidden_states" => encoder_hidden_states,
            "use_cache_branch" => use_cache_branch,
        ])?;
        let logits = outputs["logits"].extract_tensor::<f32>()?;
        let mut finished = 0;
        for (batch, tokens) in tokens.iter_mut().enumerate() {
            if tokens.len() > 1 && *tokens.last().unwrap() == 50256 {
                finished += 1;
                continue;
            }
            let mut probabilities = logits
                .view()
                .slice(ndarray::s![batch, -1, ..])
                .insert_axis(ndarray::Axis(0))
                .to_owned()
                .iter()
                .cloned()
                .enumerate()
                .collect::<Vec<_>>();
            probabilities
                .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));
            let token = probabilities[0].0;
            tokens.push(token as _);
        }
        Ok(finished == tokens.len())
    }

    pub fn decode(&self, last_hidden_state: &Tensor) -> Result<Vec<Vec<u32>>> {
        let num_batches = last_hidden_state.slice(ndarray::s![.., 0, 0]).len();
        let max_tokens = 50;
        let mut tokens = vec![Vec::with_capacity(max_tokens); num_batches];
        for i in 0..num_batches {
            tokens[i].push(50256);
        }
        for _ in 0..max_tokens {
            if self.decode_one(&last_hidden_state, &mut tokens)? {
                break;
            }
        }
        Ok(tokens)
    }

    pub fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        Ok(self
            .tokenizer
            .decode(tokens, true)
            .unwrap()
            .trim()
            .to_string())
    }

    pub fn infer(&self, batch: &[RgbImage]) -> Result<Vec<String>> {
        let input = self.prepare(batch);
        let last_hidden_state = self.encode(&input)?;
        let mut batch = Vec::with_capacity(batch.len());
        for tokens in self.decode(&last_hidden_state)? {
            batch.push(self.detokenize(&tokens)?);
        }
        Ok(batch)
    }

    fn process_batch(&self, batch: &[RgbImage], outputs: &[PathBuf]) -> Result<()> {
        let results = self.infer(batch)?;
        for (result, output) in results.iter().zip(outputs) {
            let mut w = BufWriter::new(
                OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(output)?,
            );
            w.write_all(result.as_bytes())?;
        }
        Ok(())
    }

    pub fn image_to_text(&self, inputs: &[PathBuf], output: &Path) -> Result<()> {
        let mut batch = Vec::with_capacity(self.batch_size);
        let mut outputs = Vec::with_capacity(self.batch_size);
        for input in inputs {
            let pixels = self.read_image(input)?;
            let basename = input
                .file_stem()
                .context("invalid input")?
                .to_str()
                .context("invalid input")?;
            let output = output.join(format!("{basename}.txt"));
            batch.push(pixels);
            outputs.push(output);
            if batch.len() == self.batch_size {
                self.process_batch(&batch, &outputs)?;
                batch.clear();
                outputs.clear();
            }
        }
        if !batch.is_empty() {
            self.process_batch(&batch, &outputs)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INPUT_IMG: &str = "example/cats.jpg";
    const INPUT_SVG: &str = "example/parrot.svg";
    const TOKENS: &[u32] = &[
        50256, 64, 3797, 16299, 319, 257, 18507, 351, 257, 3797, 16299, 319, 340, 220, 50256,
    ];
    const ENCODER_INPUT: &str = "example/encoder_input.json";
    const LAST_HIDDEN_STATE: &str = "example/decoder_input.json";
    const RESULT: &str = "a cat laying on a couch with a cat laying on it";
    const RESULT2: &str = "two cats are laying on a bed with a pillow";
    const RESULT3: &str = "a cartoon of a bird with a green hat on";

    fn read_tensor4(path: &str) -> Result<Tensor> {
        let bytes = std::fs::read(path)?;
        let data: Vec<Vec<Vec<Vec<f32>>>> = serde_json::from_slice(&bytes)?;
        let a = data.len();
        let b = data[0].len();
        let c = data[0][0].len();
        let d = data[0][0][0].len();
        let mut input = Array::zeros((a, b, c, d)).into_dyn();
        for x in 0..a {
            for y in 0..b {
                for z in 0..c {
                    for v in 0..d {
                        input[[x, y, z, v]] = data[x][y][z][v];
                    }
                }
            }
        }
        Ok(input)
    }

    fn read_tensor3(path: &str) -> Result<Tensor> {
        let bytes = std::fs::read(path)?;
        let data: Vec<Vec<Vec<f32>>> = serde_json::from_slice(&bytes)?;
        let a = data.len();
        let b = data[0].len();
        let c = data[0][0].len();
        let mut input = Array::zeros((a, b, c)).into_dyn();
        for x in 0..a {
            for y in 0..b {
                for z in 0..c {
                    input[[x, y, z]] = data[x][y][z];
                }
            }
        }
        Ok(input)
    }

    #[test]
    fn test_detokenize() -> Result<()> {
        let vitter = Vitter::default()?;
        let result = vitter.detokenize(TOKENS)?;
        assert_eq!(result, RESULT);
        Ok(())
    }

    #[test]
    fn test_decode_one() -> Result<()> {
        let vitter = Vitter::default()?;
        let input = read_tensor3(LAST_HIDDEN_STATE)?;
        let mut tokens = vec![vec![50256]];
        vitter.decode_one(&input, &mut tokens)?;
        assert_eq!(tokens[0][1], TOKENS[1]);
        Ok(())
    }

    #[test]
    fn test_decode() -> Result<()> {
        let vitter = Vitter::default()?;
        let input = read_tensor3(LAST_HIDDEN_STATE)?;
        let tokens = vitter.decode(&input)?;
        assert_eq!(tokens[0], TOKENS);
        Ok(())
    }

    #[test]
    fn test_encode() -> Result<()> {
        let vitter = Vitter::default()?;
        let input = read_tensor4(ENCODER_INPUT)?;
        let output = vitter.encode(&input)?;
        let expected_output = read_tensor3(LAST_HIDDEN_STATE)?;
        assert_eq!(output.shape(), expected_output.shape());
        Ok(())
    }

    #[test]
    fn test_prepare() -> Result<()> {
        let vitter = Vitter::default()?;
        let pixels = vitter.read_image(INPUT_IMG.as_ref())?;
        let input = vitter.prepare(&[pixels]);
        let expected_input = read_tensor4(ENCODER_INPUT)?;
        assert_eq!(input.shape(), expected_input.shape());
        Ok(())
    }

    #[test]
    fn test_infer() -> Result<()> {
        let vitter = Vitter::default()?;
        let pixels = vitter.read_image(INPUT_IMG.as_ref())?;
        let result = vitter.infer(&[pixels])?;
        assert_eq!(result[0], RESULT2);
        Ok(())
    }

    #[test]
    fn test_svg() -> Result<()> {
        let vitter = Vitter::default()?;
        let pixels = vitter.read_svg(INPUT_SVG.as_ref())?;
        let result = vitter.infer(&[pixels])?;
        assert_eq!(result[0], RESULT3);
        Ok(())
    }
}

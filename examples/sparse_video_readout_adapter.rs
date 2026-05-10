use anyhow::{Result, ensure};
use burn_autogaze::{
    AutoGazeConfig, AutoGazeGenerateOutput, SparseReadoutGrid, SparseReadoutOptions,
    SparseVideoPatchGeometry, SparseVideoReadoutOptions, SparseVideoReadoutProjection,
    generated_to_video_readout_coords, generated_to_video_readout_tokens,
};

const FRAMES: usize = 2;
const HEIGHT: usize = 224;
const WIDTH: usize = 224;
const TUBELET_SIZE: usize = 2;
const PATCH_SIZE: usize = 16;
const CONTEXT_TOKENS: usize = 16;

fn main() -> Result<()> {
    let config = multiscale_autogaze_config();
    let generated = synthetic_generated_output(&config);
    let projection = SparseVideoReadoutProjection::from_patch_geometry(
        SparseReadoutGrid::new(HEIGHT / PATCH_SIZE, WIDTH / PATCH_SIZE),
        SparseVideoPatchGeometry::square_patch(FRAMES, HEIGHT, WIDTH, TUBELET_SIZE, PATCH_SIZE),
    )?
    .with_readout_options(
        SparseReadoutOptions::default()
            .with_max_fixations_per_frame(8)
            .with_dilation(1)
            .with_max_tokens_per_frame(CONTEXT_TOKENS),
    )
    .with_video_options(
        SparseVideoReadoutOptions::default()
            .with_tubelet_size(TUBELET_SIZE)
            .with_exact_tokens(CONTEXT_TOKENS),
    );

    let context_tokens = generated_to_video_readout_tokens(
        &generated,
        &config,
        0,
        projection.image_grid,
        projection.video_grid,
        projection.readout_options,
        projection.video_options,
    )?;
    let context_coords = generated_to_video_readout_coords(
        &generated,
        &config,
        0,
        projection.image_grid,
        projection.video_grid,
        projection.readout_options,
        projection.video_options,
    )?;
    let context_mask = DownstreamSparseTokenMask::new(
        context_tokens.clone(),
        projection.video_grid.token_count(),
    )?;

    println!("context_tokens={context_tokens:?}");
    println!("context_coords={context_coords:?}");
    println!(
        "downstream_mask dense_len={} selected={}",
        context_mask.dense_len,
        context_mask.indices.len()
    );
    Ok(())
}

fn multiscale_autogaze_config() -> AutoGazeConfig {
    let mut config = AutoGazeConfig {
        scales: "32+64+112+224".to_string(),
        num_vision_tokens_each_frame: 265,
        ..Default::default()
    };
    config.gaze_model_config.num_vision_tokens_each_frame = config.num_vision_tokens_each_frame;
    config
}

fn synthetic_generated_output(config: &AutoGazeConfig) -> AutoGazeGenerateOutput {
    let per_frame = config.num_vision_tokens_each_frame as i64;
    AutoGazeGenerateOutput {
        gazing_pos: vec![vec![
            0,
            4,
            20,
            per_frame + 8,
            per_frame + 69,
            per_frame + 120,
        ]],
        num_gazing_each_frame: vec![3, 3],
        if_padded_gazing: vec![vec![false; 6]],
        confidences: vec![vec![0.98, 0.94, 0.91, 0.97, 0.93, 0.90]],
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DownstreamSparseTokenMask {
    indices: Vec<usize>,
    dense_len: usize,
}

impl DownstreamSparseTokenMask {
    fn new(mut indices: Vec<usize>, dense_len: usize) -> Result<Self> {
        ensure!(
            dense_len > 0,
            "downstream dense token count must be nonzero"
        );
        indices.sort_unstable();
        indices.dedup();
        ensure!(
            indices.iter().all(|&index| index < dense_len),
            "downstream sparse token index outside dense token count"
        );
        Ok(Self { indices, dense_len })
    }
}

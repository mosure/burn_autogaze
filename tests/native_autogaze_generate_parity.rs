#![cfg(feature = "ndarray")]

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_autogaze::{
    AutoGazeConfig, AutoGazeGenerateOutput, AutoGazeInferenceMode, AutoGazePipeline,
    AutoGazeRgbaClipShape, AutoGazeStreamingCache, AutoGazeVisualizationMode,
    AutoGazeVisualizationState, DEFAULT_BLEND_ALPHA, FrameFixationTrace, NativeAutoGazeModel,
    fixation_alpha_mask, fixation_effective_alpha_mask, rgba_clip_to_processor_tensor,
};
use safetensors::SafeTensors;
use serde::Deserialize;
use std::fs;
use std::path::Path;

type TestBackend = burn::backend::NdArray<f32>;

#[derive(Clone, Debug)]
struct GenerateFixtureCase {
    name: String,
    dir: String,
    task_loss_requirement: Option<f32>,
}

impl GenerateFixtureCase {
    fn official_square() -> Self {
        Self {
            name: "official square 224 fixture".to_string(),
            dir: "autogaze_official_generate".to_string(),
            task_loss_requirement: None,
        }
    }

    fn birds_python() -> Self {
        Self {
            name: "upstream birds non-square fixture".to_string(),
            dir: "autogaze_birds_python_generate".to_string(),
            task_loss_requirement: Some(0.7),
        }
    }

    fn root(&self) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(&self.dir)
    }
}

fn local_workspace_checkout() -> bool {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join("bevy_burn_autogaze")
        .exists()
}

fn discover_fixture_cases_with_generated_outputs() -> Vec<GenerateFixtureCase> {
    let fixtures_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    let mut cases = fs::read_dir(&fixtures_root)
        .unwrap_or_else(|err| panic!("read fixtures root {}: {err}", fixtures_root.display()))
        .filter_map(|entry| {
            let entry = entry.expect("fixture directory entry");
            let path = entry.path();
            if !path.is_dir() {
                return None;
            }
            let fixture_path = path.join("fixture_outputs.safetensors");
            if !fixture_path.exists() {
                return None;
            }
            let bytes = fs::read(&fixture_path)
                .unwrap_or_else(|err| panic!("read fixture {}: {err}", fixture_path.display()));
            let tensors = SafeTensors::deserialize(&bytes).unwrap_or_else(|err| {
                panic!("deserialize fixture {}: {err}", fixture_path.display())
            });
            let has_generate_outputs = ["gazing_pos", "num_gazing_each_frame", "if_padded_gazing"]
                .into_iter()
                .all(|name| tensors.names().contains(&name))
                && tensors
                    .names()
                    .iter()
                    .any(|name| name.starts_with("gazing_mask_"));
            if !has_generate_outputs {
                return None;
            }
            let dir = path
                .file_name()
                .and_then(|name| name.to_str())
                .expect("fixture directory name")
                .to_string();
            Some(GenerateFixtureCase {
                name: dir.replace('_', " "),
                dir,
                task_loss_requirement: None,
            })
        })
        .collect::<Vec<_>>();
    cases.sort_by(|left, right| left.dir.cmp(&right.dir));
    cases
}

#[derive(Debug, Deserialize)]
struct BirdsFixtureMetadata {
    raw_shape: Vec<usize>,
    raw_rgba_frames: Option<Vec<String>>,
    processed_shape: Vec<usize>,
    gazing_model_shape: Vec<usize>,
    video_embeds_shape: Vec<usize>,
    streaming_video_embed_max_abs_diff: f32,
    task_loss_requirement: f32,
    num_gazing_each_frame: Vec<i64>,
}

#[derive(Debug, Deserialize)]
struct FixtureLayoutMetadata {
    frames: Option<usize>,
    target_scales: Option<Vec<usize>>,
    target_patch_size: Option<usize>,
    num_vision_tokens_each_frame: Option<usize>,
    num_gazing_each_frame: Option<Vec<i64>>,
    mask_shapes: Option<Vec<Vec<usize>>>,
    mask_sums: Option<Vec<usize>>,
    mask_frame_sums: Option<Vec<Vec<usize>>>,
}

fn fixture_layout_metadata(fixture_root: &Path) -> Option<FixtureLayoutMetadata> {
    let metadata_path = fixture_root.join("metadata.json");
    if !metadata_path.exists() {
        return None;
    }
    Some(
        serde_json::from_slice(&fs::read(&metadata_path).expect("read fixture layout metadata"))
            .expect("parse fixture layout metadata"),
    )
}

fn upstream_multiscale_config_for(
    scales: &[usize],
    patch_size: usize,
    tokens: usize,
) -> AutoGazeConfig {
    let mut config = AutoGazeConfig {
        scales: scales
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("+"),
        num_vision_tokens_each_frame: tokens,
        ..Default::default()
    };
    config.gaze_model_config.input_img_size = scales.iter().copied().max().unwrap_or(224);
    config.gaze_model_config.num_vision_tokens_each_frame = tokens;
    config.gaze_model_config.vision_model_config.kernel_size = patch_size.max(1);
    config
}

fn scale_token_count(scales: &[usize], patch_size: usize) -> usize {
    let patch_size = patch_size.max(1);
    scales
        .iter()
        .map(|scale| {
            let grid = (scale / patch_size).max(1);
            grid * grid
        })
        .sum()
}

fn fixture_multiscale_config(case: &GenerateFixtureCase) -> AutoGazeConfig {
    if let Some(metadata) = fixture_layout_metadata(&case.root())
        && let Some(scales) = metadata.target_scales
        && !scales.is_empty()
    {
        let patch_size = metadata.target_patch_size.unwrap_or(16);
        let tokens = metadata
            .num_vision_tokens_each_frame
            .unwrap_or_else(|| scale_token_count(&scales, patch_size));
        return upstream_multiscale_config_for(&scales, patch_size, tokens);
    }

    upstream_multiscale_config()
}

fn fixture_mask_layouts(tensors: &SafeTensors<'_>) -> Vec<(usize, usize)> {
    let mut layouts = Vec::new();
    for scale_idx in 0.. {
        let name = format!("gazing_mask_{scale_idx}");
        if !tensors.names().contains(&name.as_str()) {
            break;
        }
        let view = tensors.tensor(&name).expect("fixture mask tensor");
        let shape = view.shape();
        assert_eq!(shape.len(), 3, "{name} must be [batch,frames,tokens]");
        let token_count = shape[2];
        layouts.push((square_token_grid(token_count), token_count));
    }
    layouts
}

fn metadata_mask_layout(fixture_root: &Path) -> Option<Vec<(usize, usize)>> {
    let metadata = fixture_layout_metadata(fixture_root)?;
    let scales = metadata.target_scales?;
    let patch_size = metadata.target_patch_size.unwrap_or(16).max(1);
    Some(
        scales
            .into_iter()
            .map(|scale| {
                let grid = (scale / patch_size).max(1);
                (grid, grid * grid)
            })
            .collect(),
    )
}

fn fixture_mask_stats(tensors: &SafeTensors<'_>) -> (Vec<Vec<usize>>, Vec<usize>, Vec<Vec<usize>>) {
    let mut shapes = Vec::new();
    let mut sums = Vec::new();
    let mut frame_sums = Vec::new();
    for scale_idx in 0.. {
        let name = format!("gazing_mask_{scale_idx}");
        if !tensors.names().contains(&name.as_str()) {
            break;
        }
        let (values, shape) = tensor_f32_vec(tensors, &name);
        assert_eq!(shape.len(), 3, "{name} must be [batch,frames,tokens]");
        assert_eq!(shape[0], 1, "{name} fixture batch must be one");
        let mut per_frame = Vec::with_capacity(shape[1]);
        let mut total = 0usize;
        for frame_idx in 0..shape[1] {
            let start = frame_idx * shape[2];
            let end = start + shape[2];
            let count = values[start..end]
                .iter()
                .filter(|&&value| value > 0.5)
                .count();
            total += count;
            per_frame.push(count);
        }
        shapes.push(shape);
        sums.push(total);
        frame_sums.push(per_frame);
    }
    (shapes, sums, frame_sums)
}

fn assert_fixture_generated_integrity(
    case: &GenerateFixtureCase,
    fixture_root: &Path,
    tensors: &SafeTensors<'_>,
    generated: &AutoGazeGenerateOutput,
    config: &AutoGazeConfig,
    expected_layout: &[(usize, usize)],
) {
    let (mask_shapes, mask_sums, mask_frame_sums) = fixture_mask_stats(tensors);
    assert_eq!(
        expected_layout
            .iter()
            .map(|(_, tokens)| *tokens)
            .sum::<usize>(),
        config.num_vision_tokens_each_frame,
        "{}: configured vision-token count must match fixture scale layout",
        case.name
    );
    assert_eq!(
        config.gaze_model_config.num_vision_tokens_each_frame, config.num_vision_tokens_each_frame,
        "{}: top-level and gaze-model token counts must stay aligned",
        case.name
    );

    if let Some(frame_count) = mask_shapes.first().map(|shape| shape[1]) {
        assert_eq!(
            generated.num_gazing_each_frame.len(),
            frame_count,
            "{}: generated frame-count metadata must match mask tensors",
            case.name
        );
        for (shape, (_, expected_tokens)) in mask_shapes.iter().zip(expected_layout) {
            assert_eq!(
                shape[1], frame_count,
                "{}: mask frame count drifted",
                case.name
            );
            assert_eq!(
                shape[2], *expected_tokens,
                "{}: mask tensor token count drifted",
                case.name
            );
        }
    }

    if let Some(metadata) = fixture_layout_metadata(fixture_root) {
        if let Some(frames) = metadata.frames
            && let Some(shape) = mask_shapes.first()
        {
            assert_eq!(
                shape[1], frames,
                "{}: metadata frame count drifted",
                case.name
            );
        }
        if let Some(num_gazing_each_frame) = metadata.num_gazing_each_frame {
            assert_eq!(
                generated
                    .num_gazing_each_frame
                    .iter()
                    .map(|value| *value as i64)
                    .collect::<Vec<_>>(),
                num_gazing_each_frame,
                "{}: metadata per-frame gaze counts drifted",
                case.name
            );
        }
        if let Some(expected_mask_shapes) = metadata.mask_shapes {
            assert_eq!(
                mask_shapes, expected_mask_shapes,
                "{}: metadata mask shapes drifted",
                case.name
            );
        }
        if let Some(expected_mask_sums) = metadata.mask_sums {
            assert_eq!(
                mask_sums, expected_mask_sums,
                "{}: metadata mask sums drifted",
                case.name
            );
        }
        if let Some(expected_frame_sums) = metadata.mask_frame_sums {
            assert_eq!(
                mask_frame_sums, expected_frame_sums,
                "{}: metadata per-frame mask sums drifted",
                case.name
            );
        }
    }

    assert_eq!(
        generated.gazing_pos.len(),
        1,
        "{}: fixture integrity checks expect one batch",
        case.name
    );
    assert_eq!(
        generated.if_padded_gazing.len(),
        1,
        "{}: fixture integrity checks expect one padding batch",
        case.name
    );
    assert_eq!(
        generated.gazing_pos[0].len(),
        generated.if_padded_gazing[0].len(),
        "{}: generated token and padding vectors must share length",
        case.name
    );

    let valid_tokens = generated.if_padded_gazing[0]
        .iter()
        .filter(|&&padded| !padded)
        .count();
    assert_eq!(
        mask_sums.iter().sum::<usize>(),
        valid_tokens,
        "{}: fixture masks must cover every non-padded generated token exactly once",
        case.name
    );

    let mut cursor = 0usize;
    for (frame_idx, &frame_count) in generated.num_gazing_each_frame.iter().enumerate() {
        let end = cursor + frame_count;
        assert!(
            end <= generated.gazing_pos[0].len(),
            "{}: per-frame gaze counts exceed generated token vector length",
            case.name
        );
        let frame_start = (frame_idx * config.num_vision_tokens_each_frame) as i64;
        let frame_end = frame_start + config.num_vision_tokens_each_frame as i64;
        for token_idx in cursor..end {
            if generated.if_padded_gazing[0][token_idx] {
                continue;
            }
            let token = generated.gazing_pos[0][token_idx];
            assert!(
                (frame_start..frame_end).contains(&token),
                "{}: non-padded generated token {token} at index {token_idx} is outside frame {frame_idx} range [{frame_start},{frame_end})",
                case.name
            );
        }
        cursor = end;
    }
    assert_eq!(
        cursor,
        generated.gazing_pos[0].len(),
        "{}: per-frame gaze counts must account for the full generated token vector",
        case.name
    );
}

fn tensor_f32_5<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Tensor<B, 5> {
    let view = tensors.tensor(name).expect("fixture tensor");
    let shape = view.shape();
    let data: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect();
    Tensor::from_data(
        TensorData::new(data, [shape[0], shape[1], shape[2], shape[3], shape[4]]),
        device,
    )
}

fn tensor_f32_vec(tensors: &SafeTensors<'_>, name: &str) -> (Vec<f32>, Vec<usize>) {
    let view = tensors.tensor(name).expect("fixture tensor");
    let shape = view.shape().to_vec();
    let data = view
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect();
    (data, shape)
}

fn tensor_u8_vec(tensors: &SafeTensors<'_>, name: &str) -> (Vec<u8>, Vec<usize>) {
    let view = tensors.tensor(name).expect("fixture tensor");
    (view.data().to_vec(), view.shape().to_vec())
}

fn tensor_i64_vec(tensors: &SafeTensors<'_>, name: &str) -> (Vec<i64>, Vec<usize>) {
    let view = tensors.tensor(name).expect("fixture tensor");
    let shape = view.shape().to_vec();
    let data = view
        .data()
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("i64 chunk")))
        .collect();
    (data, shape)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FixtureMaskProjection {
    NativeScale,
    EffectiveUpdate,
}

fn fixture_raw_rgba(
    tensors: &SafeTensors<'_>,
    fixture_root: &Path,
) -> Option<(Vec<u8>, Vec<usize>)> {
    if tensors.names().contains(&"raw_rgba") {
        return Some(tensor_u8_vec(tensors, "raw_rgba"));
    }

    let mut frames = Vec::new();
    for frame_idx in 0.. {
        let path = fixture_root.join(format!("raw_rgba_frame_{frame_idx:02}.png"));
        if !path.exists() {
            break;
        }
        let frame = image::open(&path)
            .unwrap_or_else(|err| panic!("failed to read fixture frame {}: {err}", path.display()))
            .to_rgba8();
        frames.push(frame);
    }
    let first = frames.first()?;
    let width = first.width() as usize;
    let height = first.height() as usize;
    let mut rgba = Vec::with_capacity(frames.len() * width * height * 4);
    for frame in frames {
        assert_eq!(
            (frame.width() as usize, frame.height() as usize),
            (width, height),
            "fixture raw RGBA frames must share dimensions"
        );
        rgba.extend_from_slice(frame.as_raw());
    }
    let frame_count = rgba.len() / (width * height * 4);
    Some((rgba, vec![frame_count, height, width, 4]))
}

fn tensor_f32_4<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Tensor<B, 4> {
    let view = tensors.tensor(name).expect("fixture tensor");
    let shape = view.shape();
    let data: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect();
    Tensor::from_data(
        TensorData::new(data, [shape[0], shape[1], shape[2], shape[3]]),
        device,
    )
}

fn tensor_i64_2<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let view = tensors.tensor(name).expect("fixture tensor");
    let shape = view.shape();
    let data: Vec<i64> = view
        .data()
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("i64 chunk")))
        .collect();
    Tensor::from_data(TensorData::new(data, [shape[0], shape[1]]), device)
}

fn tensor_i64_1<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let view = tensors.tensor(name).expect("fixture tensor");
    let shape = view.shape();
    let data: Vec<i64> = view
        .data()
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("i64 chunk")))
        .collect();
    Tensor::from_data(TensorData::new(data, [shape[0]]), device)
}

fn assert_native_autogaze_generate_matches_fixture(case: GenerateFixtureCase) {
    let fixture_root = case.root();
    let fixture_path = fixture_root.join("fixture_outputs.safetensors");
    if !fixture_path.exists() {
        eprintln!(
            "skipping AutoGaze generation parity: missing committed fixture {}",
            fixture_path.display()
        );
        return;
    }
    let hf_root = Path::new(
        "/home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a",
    );
    if !hf_root.exists() {
        eprintln!(
            "skipping AutoGaze generation parity: missing Hugging Face snapshot {}",
            hf_root.display()
        );
        return;
    }

    let device = Default::default();
    let model = NativeAutoGazeModel::<TestBackend>::from_hf_dir(hf_root, &device)
        .expect("load native autogaze model");

    let bytes = fs::read(&fixture_path).expect("read fixture");
    let tensors = SafeTensors::deserialize(&bytes).expect("deserialize fixture");
    let video = tensor_f32_5::<TestBackend>(&tensors, "video", &device);
    if tensors.names().contains(&"gazing_model_video") {
        assert_preprocessing_and_embedding_stages_match_fixture(
            &model,
            &tensors,
            &fixture_root,
            video.clone(),
            &case.name,
        );
    }
    let expected_gazing_pos = tensor_i64_2::<TestBackend>(&tensors, "gazing_pos", &device)
        .into_data()
        .to_vec::<i64>()
        .expect("gazing_pos vec");
    let expected_num_gazing_each_frame =
        tensor_i64_1::<TestBackend>(&tensors, "num_gazing_each_frame", &device)
            .into_data()
            .to_vec::<i64>()
            .expect("num_gazing_each_frame vec");
    let expected_if_padded = tensor_i64_2::<TestBackend>(&tensors, "if_padded_gazing", &device)
        .into_data()
        .to_vec::<i64>()
        .expect("if_padded vec");
    let max_gaze_tokens_each_frame = expected_num_gazing_each_frame
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .max(1) as usize;

    let actual_uncached = model.gazing_model.generate_uncached(
        video.clone(),
        max_gaze_tokens_each_frame,
        case.task_loss_requirement,
    );
    let actual_cached = model.gazing_model.generate_cached(
        video.clone(),
        max_gaze_tokens_each_frame,
        case.task_loss_requirement,
    );
    let actual = model.generate_with_task_loss_requirement(
        video.clone(),
        max_gaze_tokens_each_frame,
        case.task_loss_requirement,
    );
    assert_eq!(
        actual.gazing_pos.len(),
        1,
        "{}: expected single batch fixture",
        case.name
    );
    assert_eq!(
        actual.gazing_pos, actual_uncached.gazing_pos,
        "{}: public and full-sequence gaze token ids diverged",
        case.name
    );
    assert_eq!(
        actual.num_gazing_each_frame, actual_uncached.num_gazing_each_frame,
        "{}: public and full-sequence per-frame lengths diverged",
        case.name
    );
    assert_eq!(
        actual.if_padded_gazing, actual_uncached.if_padded_gazing,
        "{}: public and full-sequence padding masks diverged",
        case.name
    );
    assert_eq!(
        actual_cached.gazing_pos, actual_uncached.gazing_pos,
        "{}: KV-cache and full-sequence gaze token ids diverged",
        case.name
    );
    assert_eq!(
        actual_cached.num_gazing_each_frame, actual_uncached.num_gazing_each_frame,
        "{}: KV-cache and full-sequence per-frame lengths diverged",
        case.name
    );
    assert_eq!(
        actual_cached.if_padded_gazing, actual_uncached.if_padded_gazing,
        "{}: KV-cache and full-sequence padding masks diverged",
        case.name
    );
    assert_one_frame_streaming_matches_cached(
        &model,
        video.clone(),
        max_gaze_tokens_each_frame,
        case.task_loss_requirement,
        &actual_cached,
        &case.name,
    );
    assert_eq!(
        actual.gazing_pos[0], expected_gazing_pos,
        "{}: generated gaze token ids diverged",
        case.name
    );
    assert_eq!(
        actual
            .num_gazing_each_frame
            .iter()
            .map(|value| *value as i64)
            .collect::<Vec<_>>(),
        expected_num_gazing_each_frame,
        "{}: per-frame gaze lengths diverged",
        case.name
    );
    assert_eq!(
        actual.if_padded_gazing[0]
            .iter()
            .map(|flag| if *flag { 1_i64 } else { 0_i64 })
            .collect::<Vec<_>>(),
        expected_if_padded,
        "{}: padding mask diverged",
        case.name
    );

    let actual_scale_masks = actual.scale_token_masks(&model.config);
    assert_eq!(
        actual_scale_masks.len(),
        1,
        "{}: expected one batch of scale masks",
        case.name
    );
    assert_eq!(
        actual_scale_masks[0].len(),
        4,
        "{}: expected the official four-scale AutoGaze layout",
        case.name
    );
    for (scale_idx, actual_mask) in actual_scale_masks[0].iter().enumerate() {
        let (expected, shape) = tensor_f32_vec(&tensors, &format!("gazing_mask_{scale_idx}"));
        assert_eq!(shape[0], 1, "{}: fixture batch must stay one", case.name);
        assert_eq!(
            shape[1],
            actual.num_gazing_each_frame.len(),
            "{}: fixture frame count must match generated masks",
            case.name
        );
        assert_eq!(
            shape[2], actual_mask.token_count,
            "{}: fixture scale token count diverged",
            case.name
        );
        let actual = actual_mask
            .frames
            .iter()
            .flat_map(|frame| frame.iter().copied())
            .collect::<Vec<_>>();
        let expected = expected
            .iter()
            .copied()
            .map(|value| value > 0.5)
            .collect::<Vec<_>>();
        assert_eq!(
            actual, expected,
            "{}: decoded per-scale mask diverged for scale {scale_idx}",
            case.name
        );
    }

    if let Some((raw_rgba, raw_shape)) = fixture_raw_rgba(&tensors, &fixture_root) {
        let realtime_budget = max_gaze_tokens_each_frame.clamp(1, 10);
        let expected_traces = model.trace_video_with_task_loss_requirement(
            video,
            realtime_budget,
            realtime_budget,
            case.task_loss_requirement,
        );
        let pipeline = AutoGazePipeline::new(model.clone())
            .with_max_gaze_tokens_each_frame(realtime_budget)
            .with_task_loss_requirement(case.task_loss_requirement);
        let actual_traces = pipeline
            .trace_rgba_clip_with_mode(
                &raw_rgba,
                AutoGazeRgbaClipShape::new(raw_shape[0], raw_shape[1], raw_shape[2]),
                realtime_budget,
                AutoGazeInferenceMode::ResizeToModelInput,
                &device,
            )
            .expect("trace raw RGBA through public pipeline");
        assert_traces_match(
            &actual_traces,
            &expected_traces,
            1.0e-5,
            &format!(
                "{}: public raw RGBA pipeline trace diverged from fixture tensor trace",
                case.name
            ),
        );
    }
}

fn upstream_multiscale_config() -> AutoGazeConfig {
    let mut config = AutoGazeConfig {
        scales: "32+64+112+224".to_string(),
        num_vision_tokens_each_frame: 265,
        ..Default::default()
    };
    config.gaze_model_config.num_vision_tokens_each_frame = 265;
    config
}

fn generated_output_from_fixture(tensors: &SafeTensors<'_>) -> AutoGazeGenerateOutput {
    let (gazing_pos, gazing_pos_shape) = tensor_i64_vec(tensors, "gazing_pos");
    let (num_gazing_each_frame, _) = tensor_i64_vec(tensors, "num_gazing_each_frame");
    let (if_padded_gazing, if_padded_shape) = tensor_i64_vec(tensors, "if_padded_gazing");

    assert_eq!(
        gazing_pos_shape.len(),
        2,
        "fixture gazing_pos must be [batch,tokens]"
    );
    assert_eq!(
        if_padded_shape, gazing_pos_shape,
        "fixture padding mask must share gazing_pos shape"
    );

    let batch = gazing_pos_shape[0];
    let tokens = gazing_pos_shape[1];
    let mut batch_tokens = Vec::with_capacity(batch);
    let mut batch_padding = Vec::with_capacity(batch);
    let mut batch_confidence = Vec::with_capacity(batch);
    for batch_idx in 0..batch {
        let start = batch_idx * tokens;
        let end = start + tokens;
        batch_tokens.push(gazing_pos[start..end].to_vec());
        batch_padding.push(
            if_padded_gazing[start..end]
                .iter()
                .map(|flag| *flag != 0)
                .collect(),
        );
        batch_confidence.push(vec![1.0; tokens]);
    }

    AutoGazeGenerateOutput {
        gazing_pos: batch_tokens,
        num_gazing_each_frame: num_gazing_each_frame
            .into_iter()
            .map(|value| value.max(0) as usize)
            .collect(),
        if_padded_gazing: batch_padding,
        confidences: batch_confidence,
    }
}

fn fixture_expected_alpha_from_scale_masks(
    tensors: &SafeTensors<'_>,
    width: usize,
    height: usize,
    frame_idx: usize,
    projection: FixtureMaskProjection,
) -> Vec<u8> {
    let mut selected = Vec::<(usize, usize, usize)>::new();
    for scale_idx in 0.. {
        let name = format!("gazing_mask_{scale_idx}");
        if !tensors.names().contains(&name.as_str()) {
            break;
        }
        let (values, shape) = tensor_f32_vec(tensors, &name);
        assert_eq!(shape.len(), 3, "{name} must be [batch,frames,tokens]");
        assert_eq!(shape[0], 1, "{name} fixture batch must be one");
        assert!(
            frame_idx < shape[1],
            "{name} missing fixture frame {frame_idx}"
        );
        let grid = square_token_grid(shape[2]);
        let frame_start = frame_idx * shape[2];
        for token in 0..shape[2] {
            if values[frame_start + token] > 0.5 {
                selected.push((grid, token / grid, token % grid));
            }
        }
    }

    let effective_grid = selected
        .iter()
        .map(|(grid, _, _)| *grid)
        .max()
        .unwrap_or(14)
        .max(14);
    let mut alpha = vec![0u8; width.max(1) * height.max(1)];
    for (grid, row, col) in selected {
        let (rect_grid, rect_row, rect_col) = match projection {
            FixtureMaskProjection::NativeScale => (grid, row, col),
            FixtureMaskProjection::EffectiveUpdate => {
                let center_x = (col as f64 + 0.5) / grid as f64;
                let center_y = (row as f64 + 0.5) / grid as f64;
                let col = (center_x.clamp(0.0, 1.0 - f64::EPSILON) * effective_grid as f64).floor()
                    as usize;
                let row = (center_y.clamp(0.0, 1.0 - f64::EPSILON) * effective_grid as f64).floor()
                    as usize;
                (effective_grid, row, col)
            }
        };
        fill_expected_alpha_cell(&mut alpha, width, height, rect_grid, rect_row, rect_col);
    }
    alpha
}

fn fill_expected_alpha_cell(
    alpha: &mut [u8],
    width: usize,
    height: usize,
    grid: usize,
    row: usize,
    col: usize,
) {
    let grid = grid.max(1);
    let (x0, x1) = fixture_pixel_range(
        col as f64 / grid as f64,
        (col + 1) as f64 / grid as f64,
        width,
    );
    let (y0, y1) = fixture_pixel_range(
        row as f64 / grid as f64,
        (row + 1) as f64 / grid as f64,
        height,
    );
    for y in y0..y1 {
        let start = y * width + x0;
        let end = y * width + x1;
        alpha[start..end].fill(255);
    }
}

fn fixture_pixel_range(min: f64, max: f64, extent: usize) -> (usize, usize) {
    let extent = extent.max(1);
    let extent_f = extent as f64;
    let mut start = (min.clamp(0.0, 1.0) * extent_f).floor() as usize;
    let mut end = (max.clamp(0.0, 1.0) * extent_f).ceil() as usize;
    start = start.min(extent.saturating_sub(1));
    end = end.min(extent);
    if end <= start {
        end = (start + 1).min(extent);
    }
    (start, end)
}

fn square_token_grid(tokens: usize) -> usize {
    let grid = (tokens as f64).sqrt() as usize;
    assert_eq!(grid * grid, tokens, "fixture token count must be square");
    grid
}

fn colored_mask_footprint(mask_rgba: &[u8]) -> Vec<u8> {
    mask_rgba
        .chunks_exact(4)
        .map(|pixel| {
            if pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0 {
                255
            } else {
                0
            }
        })
        .collect()
}

fn alpha_pixel_count(alpha: &[u8]) -> usize {
    alpha.iter().filter(|&&value| value != 0).count()
}

fn copy_expected_interframe_update(source: &[u8], target: &mut [u8], alpha: &[u8]) {
    assert_eq!(source.len(), target.len());
    assert_eq!(source.len(), alpha.len() * 4);
    for (pixel_idx, &mask) in alpha.iter().enumerate() {
        if mask == 0 {
            continue;
        }
        let offset = pixel_idx * 4;
        target[offset..offset + 4].copy_from_slice(&source[offset..offset + 4]);
    }
}

fn assert_fixture_generated_masks_decode_without_model_snapshot(case: GenerateFixtureCase) {
    let fixture_root = case.root();
    let fixture_path = fixture_root.join("fixture_outputs.safetensors");
    let bytes = fs::read(&fixture_path)
        .unwrap_or_else(|err| panic!("read fixture {}: {err}", fixture_path.display()));
    let tensors = SafeTensors::deserialize(&bytes).expect("deserialize fixture");
    let generated = generated_output_from_fixture(&tensors);
    let config = fixture_multiscale_config(&case);
    let masks = generated.scale_token_masks(&config);
    let expected_layout = fixture_mask_layouts(&tensors);
    assert_fixture_generated_integrity(
        &case,
        &fixture_root,
        &tensors,
        &generated,
        &config,
        &expected_layout,
    );

    assert_eq!(
        masks.len(),
        1,
        "{}: fixtures currently cover a single generated batch",
        case.name
    );
    assert_eq!(
        masks[0]
            .iter()
            .map(|mask| (mask.grid, mask.token_count))
            .collect::<Vec<_>>(),
        expected_layout,
        "{}: decoded upstream multi-scale layout drifted",
        case.name
    );
    if let Some(metadata_layout) = metadata_mask_layout(&fixture_root) {
        assert_eq!(
            metadata_layout, expected_layout,
            "{}: fixture mask layout diverged from metadata target scales",
            case.name
        );
    }

    for (scale_idx, actual_mask) in masks[0].iter().enumerate() {
        let (expected, shape) = tensor_f32_vec(&tensors, &format!("gazing_mask_{scale_idx}"));
        assert_eq!(shape[0], 1, "{}: fixture batch must stay one", case.name);
        assert_eq!(
            shape[1],
            generated.num_gazing_each_frame.len(),
            "{}: fixture frame count must match decoded masks",
            case.name
        );
        assert_eq!(
            shape[2], actual_mask.token_count,
            "{}: scale {scale_idx} token count drifted",
            case.name
        );
        let actual = actual_mask
            .frames
            .iter()
            .flat_map(|frame| frame.iter().copied())
            .collect::<Vec<_>>();
        let expected = expected
            .iter()
            .copied()
            .map(|value| value > 0.5)
            .collect::<Vec<_>>();
        assert_eq!(
            actual, expected,
            "{}: fixture-only per-scale mask decode diverged for scale {scale_idx}",
            case.name
        );
    }
}

fn assert_preprocessing_and_embedding_stages_match_fixture(
    model: &NativeAutoGazeModel<TestBackend>,
    tensors: &SafeTensors<'_>,
    fixture_root: &Path,
    video: Tensor<TestBackend, 5>,
    case_name: &str,
) {
    let device = Default::default();
    if let Some((raw_rgba, raw_shape)) = fixture_raw_rgba(tensors, fixture_root) {
        assert_eq!(
            raw_shape.len(),
            4,
            "{case_name}: raw_rgba fixture must be [time,height,width,rgba]"
        );
        assert_eq!(
            raw_shape[3], 4,
            "{case_name}: raw_rgba fixture must contain RGBA channels"
        );
        let actual_processor_video = rgba_clip_to_processor_tensor::<TestBackend>(
            &raw_rgba,
            AutoGazeRgbaClipShape::new(raw_shape[0], raw_shape[1], raw_shape[2]),
            &device,
        )
        .expect("raw RGBA processor tensor");
        assert_tensor_close_5(
            actual_processor_video,
            video.clone(),
            6.0e-2,
            &format!("{case_name}: raw RGBA processor path drifted from upstream Python"),
        );
    }

    let expected_gazing_video = tensor_f32_5::<TestBackend>(tensors, "gazing_model_video", &device);
    let actual_gazing_video = model.gazing_model.prepare_video(video.clone());
    assert_tensor_close_5(
        actual_gazing_video.clone(),
        expected_gazing_video,
        1.0e-4,
        &format!("{case_name}: square gazing-model input drifted from upstream"),
    );

    let expected_embeds = tensor_f32_4::<TestBackend>(tensors, "video_embeds", &device);
    let expected_streaming_embeds =
        tensor_f32_4::<TestBackend>(tensors, "streaming_video_embeds", &device);
    let (actual_embeds, _) =
        model
            .gazing_model
            .embed_video(actual_gazing_video.clone(), false, None);
    assert_tensor_close_4(
        actual_embeds,
        expected_embeds.clone(),
        2.0e-3,
        &format!("{case_name}: full-video vision embeddings drifted from upstream"),
    );

    let frames = video.shape().dims::<5>()[1];
    let mut past_conv_values = None;
    let mut streaming_embeds = Vec::with_capacity(frames);
    for frame_idx in 0..frames {
        let frame = actual_gazing_video
            .clone()
            .slice_dim(1, frame_idx..(frame_idx + 1));
        let (frame_embeds, next_past_conv_values) =
            model
                .gazing_model
                .embed_video(frame, true, past_conv_values);
        streaming_embeds.push(frame_embeds);
        past_conv_values = Some(next_past_conv_values);
    }
    let actual_streaming_embeds = Tensor::cat(streaming_embeds, 1);
    assert_tensor_close_4(
        actual_streaming_embeds,
        expected_streaming_embeds,
        2.0e-3,
        &format!("{case_name}: one-frame streaming vision embeddings drifted from upstream"),
    );
    assert_tensor_close_4(
        expected_embeds,
        tensor_f32_4::<TestBackend>(tensors, "streaming_video_embeds", &device),
        2.0e-4,
        &format!("{case_name}: upstream streaming and full-video embeddings diverged"),
    );
}

fn assert_tensor_close_5(
    actual: Tensor<TestBackend, 5>,
    expected: Tensor<TestBackend, 5>,
    tolerance: f32,
    message: &str,
) {
    let actual_shape = actual.shape().dims::<5>();
    let expected_shape = expected.shape().dims::<5>();
    assert_eq!(actual_shape, expected_shape, "{message}: shape mismatch");
    assert_vec_close(
        actual
            .into_data()
            .to_vec::<f32>()
            .expect("actual tensor values"),
        expected
            .into_data()
            .to_vec::<f32>()
            .expect("expected tensor values"),
        tolerance,
        message,
    );
}

fn assert_tensor_close_4(
    actual: Tensor<TestBackend, 4>,
    expected: Tensor<TestBackend, 4>,
    tolerance: f32,
    message: &str,
) {
    let actual_shape = actual.shape().dims::<4>();
    let expected_shape = expected.shape().dims::<4>();
    assert_eq!(actual_shape, expected_shape, "{message}: shape mismatch");
    assert_vec_close(
        actual
            .into_data()
            .to_vec::<f32>()
            .expect("actual tensor values"),
        expected
            .into_data()
            .to_vec::<f32>()
            .expect("expected tensor values"),
        tolerance,
        message,
    );
}

fn assert_vec_close(actual: Vec<f32>, expected: Vec<f32>, tolerance: f32, message: &str) {
    assert_eq!(actual.len(), expected.len(), "{message}: length mismatch");
    let mut max_diff = 0.0_f32;
    let mut max_index = 0usize;
    for (index, (left, right)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (left - right).abs();
        if diff > max_diff {
            max_diff = diff;
            max_index = index;
        }
    }
    assert!(
        max_diff <= tolerance,
        "{message}: max abs diff {max_diff} at flat index {max_index} exceeds {tolerance}"
    );
}

fn assert_one_frame_streaming_matches_cached(
    model: &NativeAutoGazeModel<TestBackend>,
    video: Tensor<TestBackend, 5>,
    max_gaze_tokens_each_frame: usize,
    task_loss_requirement: Option<f32>,
    expected: &AutoGazeGenerateOutput,
    case_name: &str,
) {
    let frames = video.shape().dims::<5>()[1];
    let mut cache = AutoGazeStreamingCache::new(frames.max(1));
    let mut gazing_pos = vec![Vec::<i64>::new()];
    let mut if_padded_gazing = vec![Vec::<bool>::new()];
    let mut confidences = vec![Vec::<f32>::new()];
    let mut num_gazing_each_frame = Vec::with_capacity(frames);

    for frame_idx in 0..frames {
        let frame = video.clone().slice_dim(1, frame_idx..(frame_idx + 1));
        let next = model.gazing_model.generate_streaming_cached(
            frame,
            &mut cache,
            max_gaze_tokens_each_frame,
            task_loss_requirement,
        );
        num_gazing_each_frame.extend(next.num_gazing_each_frame.iter().copied());
        let frame_offset = (frame_idx * model.config.num_vision_tokens_each_frame) as i64;
        gazing_pos[0].extend(next.gazing_pos[0].iter().map(|token| token + frame_offset));
        if_padded_gazing[0].extend(next.if_padded_gazing[0].iter().copied());
        confidences[0].extend(next.confidences[0].iter().copied());
    }

    let actual = AutoGazeGenerateOutput {
        gazing_pos,
        num_gazing_each_frame,
        if_padded_gazing,
        confidences,
    };
    assert_eq!(
        actual.gazing_pos, expected.gazing_pos,
        "{case_name}: one-frame streaming cache token ids diverged from full-video cached generation"
    );
    assert_eq!(
        actual.num_gazing_each_frame, expected.num_gazing_each_frame,
        "{case_name}: one-frame streaming cache per-frame lengths diverged from full-video cached generation"
    );
    assert_eq!(
        actual.if_padded_gazing, expected.if_padded_gazing,
        "{case_name}: one-frame streaming cache padding masks diverged from full-video cached generation"
    );
    for (index, (left, right)) in actual.confidences[0]
        .iter()
        .zip(expected.confidences[0].iter())
        .enumerate()
    {
        assert!(
            (left - right).abs() < 1.0e-5,
            "{case_name}: one-frame streaming confidence {index} diverged: {left} vs {right}"
        );
    }
}

fn assert_traces_match(
    actual: &[FrameFixationTrace],
    expected: &[FrameFixationTrace],
    epsilon: f32,
    message: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{message}: batch mismatch");
    for (batch_idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.frames.len(),
            expected.frames.len(),
            "{message}: frame count mismatch for batch {batch_idx}"
        );
        for (frame_idx, (actual, expected)) in
            actual.frames.iter().zip(&expected.frames).enumerate()
        {
            assert_eq!(
                actual.points.len(),
                expected.points.len(),
                "{message}: point count mismatch for batch {batch_idx} frame {frame_idx}"
            );
            for (point_idx, (actual, expected)) in
                actual.points.iter().zip(&expected.points).enumerate()
            {
                assert!(
                    (actual.x - expected.x).abs() <= epsilon
                        && (actual.y - expected.y).abs() <= epsilon
                        && (actual.cell_width() - expected.cell_width()).abs() <= epsilon
                        && (actual.cell_height() - expected.cell_height()).abs() <= epsilon,
                    "{message}: point {point_idx} diverged for batch {batch_idx} frame {frame_idx}: {actual:?} vs {expected:?}"
                );
            }
        }
    }
}

#[test]
fn native_autogaze_generate_matches_official_fixture() {
    assert_native_autogaze_generate_matches_fixture(GenerateFixtureCase::official_square());
}

#[test]
fn upstream_generated_masks_decode_without_model_snapshot() {
    let cases = discover_fixture_cases_with_generated_outputs();
    assert!(
        !cases.is_empty(),
        "expected at least the official generated fixture, found {cases:?}"
    );
    assert!(
        cases
            .iter()
            .any(|case| case.dir == "autogaze_official_generate"),
        "discovery missed official generated fixture: {cases:?}"
    );
    assert!(
        cases
            .iter()
            .any(|case| case.dir == "autogaze_upstream_resize_224"),
        "discovery missed packaged upstream seeded 224 generated fixture: {cases:?}"
    );
    assert!(
        cases
            .iter()
            .any(|case| case.dir == "autogaze_upstream_tile_448"),
        "discovery missed packaged upstream 448 generated fixture: {cases:?}"
    );
    if local_workspace_checkout() {
        assert!(
            cases
                .iter()
                .any(|case| case.dir == "autogaze_birds_python_generate"),
            "discovery missed birds generated fixture: {cases:?}"
        );
    } else if !cases
        .iter()
        .any(|case| case.dir == "autogaze_birds_python_generate")
    {
        eprintln!("skipping birds generated fixture discovery check in package checkout");
    }
    for case in cases {
        assert_fixture_generated_masks_decode_without_model_snapshot(case);
    }
}

#[test]
fn upstream_birds_visualization_matches_fixture_masks_without_model_snapshot() {
    let case = GenerateFixtureCase::birds_python();
    let fixture_root = case.root();
    let fixture_path = fixture_root.join("fixture_outputs.safetensors");
    if !fixture_path.exists() {
        if local_workspace_checkout() {
            panic!(
                "missing birds visualization fixture in workspace checkout: {}",
                fixture_path.display()
            );
        }
        eprintln!(
            "skipping birds visualization fixture in package checkout: missing {}",
            fixture_path.display()
        );
        return;
    }
    let bytes = fs::read(&fixture_path)
        .unwrap_or_else(|err| panic!("read fixture {}: {err}", fixture_path.display()));
    let tensors = SafeTensors::deserialize(&bytes).expect("deserialize fixture");
    let (raw_rgba, raw_shape) =
        fixture_raw_rgba(&tensors, &fixture_root).expect("birds raw RGBA fixture frames");
    assert_eq!(raw_shape, vec![2, 1080, 1920, 4]);

    let width = raw_shape[2];
    let height = raw_shape[1];
    let frame_bytes = width * height * 4;
    let generated = generated_output_from_fixture(&tensors);
    let config = upstream_multiscale_config();
    let traces = generated.traces(&config, 0);
    assert_eq!(traces.len(), 1);
    assert_eq!(traces[0].frames.len(), raw_shape[0]);

    let mut blend_state = AutoGazeVisualizationState::new(AutoGazeVisualizationMode::FullBlend, 30);
    for frame_idx in 0..raw_shape[0] {
        let points = &traces[0].frames[frame_idx].points;
        let native_expected = fixture_expected_alpha_from_scale_masks(
            &tensors,
            width,
            height,
            frame_idx,
            FixtureMaskProjection::NativeScale,
        );
        assert_eq!(
            fixation_alpha_mask(width, height, points, 1.0),
            native_expected,
            "fixture frame {frame_idx}: decoded points no longer match upstream native-scale masks"
        );

        let frame_start = frame_idx * frame_bytes;
        let frame = &raw_rgba[frame_start..frame_start + frame_bytes];
        let panels = blend_state
            .visualize_rgba_panels(frame, width, height, points, 1.0, DEFAULT_BLEND_ALPHA)
            .expect("visualize fixture frame");
        assert_eq!(
            colored_mask_footprint(&panels.mask_rgba),
            native_expected,
            "fixture frame {frame_idx}: visible mask panel drifted from native-scale upstream mask cells"
        );

        let effective_expected = fixture_expected_alpha_from_scale_masks(
            &tensors,
            width,
            height,
            frame_idx,
            FixtureMaskProjection::EffectiveUpdate,
        );
        let effective_actual = fixation_effective_alpha_mask(width, height, points, 1.0);
        if effective_actual != effective_expected {
            let false_positive = effective_actual
                .iter()
                .zip(&effective_expected)
                .filter(|(left, right)| **left != 0 && **right == 0)
                .count();
            let false_negative = effective_actual
                .iter()
                .zip(&effective_expected)
                .filter(|(left, right)| **left == 0 && **right != 0)
                .count();
            panic!(
                "fixture frame {frame_idx}: effective alpha drifted from upstream masks: false_positive={false_positive} false_negative={false_negative}"
            );
        }
        assert_eq!(
            panels.mask_pixel_count,
            alpha_pixel_count(&native_expected),
            "fixture frame {frame_idx}: visualization output/update footprint drifted from native-scale upstream mask cells"
        );
        assert_eq!(
            panels.updated_pixel_count, panels.mask_pixel_count,
            "full-blend output should update exactly the visible native-scale mask footprint"
        );
    }

    let mut interframe_state =
        AutoGazeVisualizationState::new(AutoGazeVisualizationMode::Interframe, 30);
    let first = &raw_rgba[..frame_bytes];
    let second = &raw_rgba[frame_bytes..frame_bytes * 2];
    let first_points = &traces[0].frames[0].points;
    let second_points = &traces[0].frames[1].points;
    let first_panels = interframe_state
        .visualize_rgba_panels(first, width, height, first_points, 1.0, DEFAULT_BLEND_ALPHA)
        .expect("visualize first interframe fixture frame");
    assert_eq!(
        first_panels.blend_rgba, first,
        "first interframe frame must be a full keyframe"
    );
    assert_eq!(first_panels.updated_pixel_count, width * height);

    let second_native = fixture_expected_alpha_from_scale_masks(
        &tensors,
        width,
        height,
        1,
        FixtureMaskProjection::NativeScale,
    );
    let mut expected_second = first.to_vec();
    copy_expected_interframe_update(second, &mut expected_second, &second_native);
    let second_panels = interframe_state
        .visualize_rgba_panels(
            second,
            width,
            height,
            second_points,
            1.0,
            DEFAULT_BLEND_ALPHA,
        )
        .expect("visualize second interframe fixture frame");
    assert_eq!(
        second_panels.blend_rgba, expected_second,
        "second interframe output must update only visible native-scale masked cells"
    );
    assert_eq!(
        second_panels.updated_pixel_count,
        alpha_pixel_count(&second_native)
    );
}

#[test]
fn native_autogaze_generate_matches_upstream_birds_fixture() {
    let fixture_root = GenerateFixtureCase::birds_python().root();
    let metadata_path = fixture_root.join("metadata.json");
    if metadata_path.exists() {
        let metadata: BirdsFixtureMetadata =
            serde_json::from_slice(&fs::read(&metadata_path).expect("read birds fixture metadata"))
                .expect("parse birds fixture metadata");
        assert_eq!(
            metadata.raw_shape,
            vec![2, 1080, 1920, 3],
            "birds fixture must retain full-resolution source-frame coverage"
        );
        if let Some(raw_rgba_frames) = &metadata.raw_rgba_frames {
            assert_eq!(raw_rgba_frames.len(), 2);
            for frame in raw_rgba_frames {
                assert!(
                    fixture_root.join(frame).exists(),
                    "metadata references missing raw RGBA frame {frame}"
                );
            }
        }
        assert_eq!(
            metadata.processed_shape,
            vec![1, 2, 3, 224, 398],
            "birds fixture must exercise the non-square upstream preprocessor path"
        );
        assert_eq!(metadata.gazing_model_shape, vec![1, 2, 3, 224, 224]);
        assert_eq!(metadata.video_embeds_shape, vec![1, 2, 196, 192]);
        assert!(
            metadata.streaming_video_embed_max_abs_diff <= 2.0e-4,
            "upstream streaming/full-video embed diff too large: {}",
            metadata.streaming_video_embed_max_abs_diff
        );
        assert_eq!(metadata.task_loss_requirement, 0.7);
        assert_eq!(metadata.num_gazing_each_frame, vec![198, 30]);
    }

    assert_native_autogaze_generate_matches_fixture(GenerateFixtureCase::birds_python());
}

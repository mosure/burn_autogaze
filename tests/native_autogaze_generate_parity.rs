#![cfg(feature = "ndarray")]

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_autogaze::NativeAutoGazeModel;
use safetensors::SafeTensors;
use std::fs;
use std::path::Path;

type TestBackend = burn::backend::NdArray<f32>;

fn fixture_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("autogaze_official_generate")
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

#[test]
fn native_autogaze_generate_matches_official_fixture() {
    let fixture_root = fixture_root();
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

    let actual_uncached =
        model
            .gazing_model
            .generate_uncached(video.clone(), max_gaze_tokens_each_frame, None);
    let actual_cached =
        model
            .gazing_model
            .generate_cached(video.clone(), max_gaze_tokens_each_frame, None);
    let actual = model.generate(video, max_gaze_tokens_each_frame);
    assert_eq!(actual.gazing_pos.len(), 1, "expected single batch fixture");
    assert_eq!(
        actual.gazing_pos, actual_uncached.gazing_pos,
        "public and full-sequence gaze token ids diverged"
    );
    assert_eq!(
        actual.num_gazing_each_frame, actual_uncached.num_gazing_each_frame,
        "public and full-sequence per-frame lengths diverged"
    );
    assert_eq!(
        actual.if_padded_gazing, actual_uncached.if_padded_gazing,
        "public and full-sequence padding masks diverged"
    );
    assert_eq!(
        actual_cached.gazing_pos, actual_uncached.gazing_pos,
        "KV-cache and full-sequence gaze token ids diverged"
    );
    assert_eq!(
        actual_cached.num_gazing_each_frame, actual_uncached.num_gazing_each_frame,
        "KV-cache and full-sequence per-frame lengths diverged"
    );
    assert_eq!(
        actual_cached.if_padded_gazing, actual_uncached.if_padded_gazing,
        "KV-cache and full-sequence padding masks diverged"
    );
    assert_eq!(
        actual.gazing_pos[0], expected_gazing_pos,
        "generated gaze token ids diverged"
    );
    assert_eq!(
        actual
            .num_gazing_each_frame
            .iter()
            .map(|value| *value as i64)
            .collect::<Vec<_>>(),
        expected_num_gazing_each_frame,
        "per-frame gaze lengths diverged"
    );
    assert_eq!(
        actual.if_padded_gazing[0]
            .iter()
            .map(|flag| if *flag { 1_i64 } else { 0_i64 })
            .collect::<Vec<_>>(),
        expected_if_padded,
        "padding mask diverged"
    );
}

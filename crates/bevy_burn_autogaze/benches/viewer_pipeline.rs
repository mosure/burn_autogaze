use bevy::{
    asset::RenderAssetUsages,
    image::ImageSampler,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use burn_autogaze::{AutoGazeVisualizationMode, AutoGazeVisualizationState, FixationPoint};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

#[derive(Clone, Copy)]
struct VideoCase {
    name: &'static str,
    width: usize,
    height: usize,
}

#[derive(Clone, Copy)]
struct VisualizationCase {
    name: &'static str,
    mode: AutoGazeVisualizationMode,
    prime_interframe: bool,
}

const VIDEO_CASES: &[VideoCase] = &[
    VideoCase {
        name: "720p",
        width: 1280,
        height: 720,
    },
    VideoCase {
        name: "1080p",
        width: 1920,
        height: 1080,
    },
];
const VISUALIZATION_CASES: &[VisualizationCase] = &[
    VisualizationCase {
        name: "full-blend",
        mode: AutoGazeVisualizationMode::FullBlend,
        prime_interframe: false,
    },
    VisualizationCase {
        name: "interframe-delta",
        mode: AutoGazeVisualizationMode::Interframe,
        prime_interframe: true,
    },
];
const BLEND_ALPHA: f32 = 0.72;
const KEYFRAME_DURATION: usize = 30;

fn bench_viewer_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("bevy_autogaze_viewer_pipeline");
    group.sample_size(10);

    for &case in VIDEO_CASES {
        group.throughput(Throughput::Bytes((case.width * case.height * 4 * 3) as u64));
        for &visualization in VISUALIZATION_CASES {
            bench_viewer_case(&mut group, case, visualization);
        }
    }

    group.finish();
}

fn bench_viewer_case(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    case: VideoCase,
    visualization: VisualizationCase,
) {
    let current = deterministic_rgba_frame(case.height, case.width, 11);
    let previous = deterministic_rgba_frame(case.height, case.width, 7);
    let points = multiscale_fixations();
    group.bench_with_input(
        BenchmarkId::new(visualization.name, case.name),
        &case,
        |b, _| {
            b.iter_batched(
                || {
                    let mut state =
                        AutoGazeVisualizationState::new(visualization.mode, KEYFRAME_DURATION);
                    if visualization.prime_interframe {
                        state
                            .visualize_rgba(
                                &previous,
                                case.width,
                                case.height,
                                &points,
                                1.0,
                                BLEND_ALPHA,
                            )
                            .expect("prime interframe state");
                    }
                    let mut images = Assets::<Image>::default();
                    let handle = images.add(empty_side_by_side_image(case));
                    (state, images, handle)
                },
                |(mut state, mut images, handle)| {
                    let output = state
                        .visualize_rgba(
                            &current,
                            case.width,
                            case.height,
                            &points,
                            1.0,
                            BLEND_ALPHA,
                        )
                        .expect("visualize autogaze frame");
                    write_side_by_side_image(
                        &handle,
                        &mut images,
                        output.side_by_side_width as u32,
                        output.height as u32,
                        output.side_by_side_rgba,
                    );
                    black_box(images.get(&handle).and_then(|image| image.data.as_ref()));
                },
                BatchSize::LargeInput,
            );
        },
    );
}

fn empty_side_by_side_image(case: VideoCase) -> Image {
    let width = (case.width * 3) as u32;
    let height = case.height as u32;
    let mut image = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::nearest();
    image
}

fn write_side_by_side_image(
    handle: &Handle<Image>,
    images: &mut Assets<Image>,
    width: u32,
    height: u32,
    rgba: Vec<u8>,
) {
    if let Some(mut image) = images.get_mut(handle)
        && image.width() == width
        && image.height() == height
        && image.texture_descriptor.format == TextureFormat::Rgba8UnormSrgb
    {
        image.data = Some(rgba);
        image.sampler = ImageSampler::nearest();
        return;
    }

    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::nearest();
    let _ = images.insert(handle.id(), image);
}

fn deterministic_rgba_frame(height: usize, width: usize, frame: usize) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(height * width * 4);
    for y in 0..height {
        for x in 0..width {
            rgba.push(((x + frame * 13) % 256) as u8);
            rgba.push(((y + frame * 29) % 256) as u8);
            rgba.push(((x + y + frame * 7) % 256) as u8);
            rgba.push(255);
        }
    }
    rgba
}

fn multiscale_fixations() -> Vec<FixationPoint> {
    vec![
        FixationPoint::with_extent(0.25, 0.25, 0.5, 0.5, 0.98),
        FixationPoint::with_extent(0.625, 0.125, 0.25, 0.25, 0.91),
        FixationPoint::with_extent(3.5 / 7.0, 5.5 / 7.0, 1.0 / 7.0, 1.0 / 7.0, 0.84),
        FixationPoint::with_extent(11.5 / 14.0, 8.5 / 14.0, 1.0 / 14.0, 1.0 / 14.0, 0.77),
    ]
}

criterion_group!(benches, bench_viewer_pipeline);
criterion_main!(benches);

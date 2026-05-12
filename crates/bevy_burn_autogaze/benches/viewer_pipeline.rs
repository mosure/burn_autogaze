use bevy::{
    asset::RenderAssetUsages,
    image::ImageSampler,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use burn_autogaze::{
    AutoGazeMaskGeometryMode, AutoGazeRgbaVisualizationBuffers, AutoGazeRgbaVisualizationOptions,
    AutoGazeVisualizationMode, AutoGazeVisualizationState,
    DEFAULT_TENSOR_FULL_FRAME_UPDATE_MIN_RATIO, FixationPoint,
};
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
    split_panels: bool,
}

#[derive(Clone, Copy)]
struct GeometryCase {
    name: &'static str,
    mode: AutoGazeMaskGeometryMode,
}

#[derive(Clone, Copy)]
enum FixationCase {
    Multiscale,
    TinySparse,
    CoarseDense,
    DenseGrid64,
    RedundantFullFrame,
}

impl FixationCase {
    const fn name(self) -> &'static str {
        match self {
            Self::Multiscale => "multiscale",
            Self::TinySparse => "tiny-sparse",
            Self::CoarseDense => "coarse-dense",
            Self::DenseGrid64 => "dense-grid-64",
            Self::RedundantFullFrame => "redundant-full-frame",
        }
    }

    fn points(self) -> Vec<FixationPoint> {
        match self {
            Self::Multiscale => multiscale_fixations(),
            Self::TinySparse => vec![FixationPoint::with_grid_extent(
                0.5 / 64.0,
                0.5 / 64.0,
                1.0 / 64.0,
                1.0 / 64.0,
                1.0,
                64,
            )],
            Self::CoarseDense => vec![FixationPoint::with_grid_extent(
                0.25, 0.25, 0.5, 0.5, 1.0, 2,
            )],
            Self::DenseGrid64 => dense_grid_fixations(64),
            Self::RedundantFullFrame => redundant_multiscale_fixations(),
        }
    }
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
        name: "full-blend-side-by-side",
        mode: AutoGazeVisualizationMode::FullBlend,
        prime_interframe: false,
        split_panels: false,
    },
    VisualizationCase {
        name: "interframe-delta-side-by-side",
        mode: AutoGazeVisualizationMode::Interframe,
        prime_interframe: true,
        split_panels: false,
    },
    VisualizationCase {
        name: "full-blend-panels",
        mode: AutoGazeVisualizationMode::FullBlend,
        prime_interframe: false,
        split_panels: true,
    },
    VisualizationCase {
        name: "interframe-delta-panels",
        mode: AutoGazeVisualizationMode::Interframe,
        prime_interframe: true,
        split_panels: true,
    },
];
const FIXATION_CASES: &[FixationCase] = &[
    FixationCase::Multiscale,
    FixationCase::TinySparse,
    FixationCase::CoarseDense,
    FixationCase::DenseGrid64,
    FixationCase::RedundantFullFrame,
];
const GEOMETRY_CASES: &[GeometryCase] = &[
    GeometryCase {
        name: "native",
        mode: AutoGazeMaskGeometryMode::Native,
    },
    GeometryCase {
        name: "deduplicated",
        mode: AutoGazeMaskGeometryMode::Deduplicated,
    },
    GeometryCase {
        name: "effective",
        mode: AutoGazeMaskGeometryMode::Effective,
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
            for &fixations in FIXATION_CASES {
                bench_viewer_case(&mut group, case, visualization, fixations);
            }
        }
    }

    group.finish();

    let mut group = c.benchmark_group("bevy_autogaze_mask_geometry");
    group.sample_size(10);
    for &case in VIDEO_CASES {
        group.throughput(Throughput::Bytes((case.width * case.height * 4 * 3) as u64));
        for &geometry in GEOMETRY_CASES {
            bench_geometry_case(&mut group, case, geometry);
        }
    }
    group.finish();
}

fn bench_viewer_case(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    case: VideoCase,
    visualization: VisualizationCase,
    fixations: FixationCase,
) {
    let current = deterministic_rgba_frame(case.height, case.width, 11);
    let previous = deterministic_rgba_frame(case.height, case.width, 7);
    let points = fixations.points();
    group.bench_with_input(
        BenchmarkId::new(
            format!("{}/{}", visualization.name, fixations.name()),
            case.name,
        ),
        &case,
        |b, _| {
            b.iter_batched(
                || {
                    let mut state =
                        AutoGazeVisualizationState::new(visualization.mode, KEYFRAME_DURATION);
                    if visualization.prime_interframe {
                        state
                            .visualize_rgba_with_options(&previous, &points, rgba_options(case))
                            .expect("prime interframe state");
                    }
                    let mut images = Assets::<Image>::default();
                    let handles = if visualization.split_panels {
                        BenchImages::Panels {
                            input: images.add(empty_panel_image(case)),
                            mask: images.add(empty_panel_image(case)),
                            output: images.add(empty_panel_image(case)),
                            buffers: AutoGazeRgbaVisualizationBuffers::default(),
                        }
                    } else {
                        BenchImages::SideBySide {
                            handle: images.add(empty_side_by_side_image(case)),
                        }
                    };
                    (state, images, handles)
                },
                |(mut state, mut images, handles)| match handles {
                    BenchImages::SideBySide { handle } => {
                        let output = state
                            .visualize_rgba_with_options(&current, &points, rgba_options(case))
                            .expect("visualize autogaze frame");
                        write_side_by_side_image(
                            &handle,
                            &mut images,
                            output.side_by_side_width as u32,
                            output.height as u32,
                            output.side_by_side_rgba,
                        );
                        black_box(images.get(&handle).and_then(|image| image.data.as_ref()));
                    }
                    BenchImages::Panels {
                        input,
                        mask,
                        output,
                        mut buffers,
                    } => {
                        let panels = state
                            .visualize_rgba_panels_with_options_into(
                                &current,
                                &points,
                                rgba_options(case),
                                &mut buffers,
                            )
                            .expect("visualize autogaze panels");
                        let update_ratio = panels.update_ratio();
                        write_panel_images(
                            PanelHandles {
                                input: &input,
                                mask: &mask,
                                output: &output,
                            },
                            &mut images,
                            case,
                            current.clone(),
                            std::mem::take(&mut buffers.mask_rgba),
                            std::mem::take(&mut buffers.blend_rgba),
                        );
                        black_box(update_ratio);
                        black_box(images.get(&output).and_then(|image| image.data.as_ref()));
                    }
                },
                BatchSize::LargeInput,
            );
        },
    );
}

fn bench_geometry_case(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    case: VideoCase,
    geometry: GeometryCase,
) {
    let current = deterministic_rgba_frame(case.height, case.width, 17);
    let previous = deterministic_rgba_frame(case.height, case.width, 13);
    let points = redundant_multiscale_fixations();
    group.bench_with_input(
        BenchmarkId::new("interframe-panels/redundant-multiscale", geometry.name),
        &case,
        |b, _| {
            b.iter_batched(
                || {
                    let mut state =
                        AutoGazeVisualizationState::new(AutoGazeVisualizationMode::Interframe, 0);
                    state
                        .visualize_rgba_panels_with_options(
                            &previous,
                            &points,
                            AutoGazeRgbaVisualizationOptions::new(
                                case.width,
                                case.height,
                                1.0,
                                BLEND_ALPHA,
                            )
                            .with_full_frame_update_policy(
                                DEFAULT_TENSOR_FULL_FRAME_UPDATE_MIN_RATIO,
                            )
                            .with_mask_geometry_mode(geometry.mode),
                        )
                        .expect("prime interframe state");
                    (state, AutoGazeRgbaVisualizationBuffers::default())
                },
                |(mut state, mut buffers)| {
                    let panels = state
                        .visualize_rgba_panels_with_options_into(
                            &current,
                            &points,
                            AutoGazeRgbaVisualizationOptions::new(
                                case.width,
                                case.height,
                                1.0,
                                BLEND_ALPHA,
                            )
                            .with_full_frame_update_policy(
                                DEFAULT_TENSOR_FULL_FRAME_UPDATE_MIN_RATIO,
                            )
                            .with_mask_geometry_mode(geometry.mode),
                            &mut buffers,
                        )
                        .expect("visualize redundant multiscale frame");
                    black_box(panels.mask_plan_stats.rect_count);
                    black_box(panels.update_ratio());
                    black_box(buffers.mask_rgba.as_slice());
                    black_box(buffers.blend_rgba.as_slice());
                },
                BatchSize::LargeInput,
            );
        },
    );
}

enum BenchImages {
    SideBySide {
        handle: Handle<Image>,
    },
    Panels {
        input: Handle<Image>,
        mask: Handle<Image>,
        output: Handle<Image>,
        buffers: AutoGazeRgbaVisualizationBuffers,
    },
}

struct PanelHandles<'a> {
    input: &'a Handle<Image>,
    mask: &'a Handle<Image>,
    output: &'a Handle<Image>,
}

fn empty_side_by_side_image(case: VideoCase) -> Image {
    let width = (case.width * 3) as u32;
    let height = case.height as u32;
    empty_image(width, height)
}

fn empty_panel_image(case: VideoCase) -> Image {
    empty_image(case.width as u32, case.height as u32)
}

fn empty_image(width: u32, height: u32) -> Image {
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

fn write_panel_images(
    handles: PanelHandles<'_>,
    images: &mut Assets<Image>,
    case: VideoCase,
    input_rgba: Vec<u8>,
    mask_rgba: Vec<u8>,
    output_rgba: Vec<u8>,
) {
    let width = case.width as u32;
    let height = case.height as u32;
    write_panel_image(handles.input, images, width, height, input_rgba);
    write_panel_image(handles.mask, images, width, height, mask_rgba);
    write_panel_image(handles.output, images, width, height, output_rgba);
}

fn write_panel_image(
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

fn rgba_options(case: VideoCase) -> AutoGazeRgbaVisualizationOptions {
    AutoGazeRgbaVisualizationOptions::new(case.width, case.height, 1.0, BLEND_ALPHA)
        .with_full_frame_update_policy(DEFAULT_TENSOR_FULL_FRAME_UPDATE_MIN_RATIO)
}

fn multiscale_fixations() -> Vec<FixationPoint> {
    vec![
        FixationPoint::with_extent(0.25, 0.25, 0.5, 0.5, 0.98),
        FixationPoint::with_extent(0.625, 0.125, 0.25, 0.25, 0.91),
        FixationPoint::with_extent(3.5 / 7.0, 5.5 / 7.0, 1.0 / 7.0, 1.0 / 7.0, 0.84),
        FixationPoint::with_extent(11.5 / 14.0, 8.5 / 14.0, 1.0 / 14.0, 1.0 / 14.0, 0.77),
    ]
}

fn redundant_multiscale_fixations() -> Vec<FixationPoint> {
    let mut points = Vec::new();
    points.extend(dense_grid_fixations(2));
    points.extend(dense_grid_fixations(4));
    points.extend(dense_grid_fixations(7));
    points.extend(dense_grid_fixations(14));
    points.extend(dense_grid_fixations(28));
    points
}

fn dense_grid_fixations(grid: usize) -> Vec<FixationPoint> {
    let extent = 1.0 / grid as f32;
    (0..grid)
        .flat_map(|row| {
            (0..grid).map(move |col| {
                FixationPoint::with_grid_extent(
                    (col as f32 + 0.5) * extent,
                    (row as f32 + 0.5) * extent,
                    extent,
                    extent,
                    1.0,
                    grid,
                )
            })
        })
        .collect()
}

criterion_group!(benches, bench_viewer_pipeline);
criterion_main!(benches);

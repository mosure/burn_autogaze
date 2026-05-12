use bevy::{
    asset::RenderAssetUsages,
    image::ImageSampler,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};
use bevy_burn::{BevyBurnHandle, BindingDirection, TransferKind};
use burn::tensor::Tensor;
use burn_autogaze::{AutoGazeMaskPlanStats, AutoGazeTensorInterframePath};

use crate::{AutoGazeBevyBackend, BevyDisplayTransfer};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum AutoGazeTextureLayout {
    #[default]
    SideBySide,
    Panels,
}

#[derive(Resource, Clone)]
pub(crate) struct AutoGazeTexture {
    pub(crate) image: Handle<Image>,
    pub(crate) input_image: Handle<Image>,
    pub(crate) mask_image: Handle<Image>,
    pub(crate) output_image: Handle<Image>,
    pub(crate) entity: Option<Entity>,
    pub(crate) side_by_side_entity: Option<Entity>,
    pub(crate) input_entity: Option<Entity>,
    pub(crate) mask_entity: Option<Entity>,
    pub(crate) output_entity: Option<Entity>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) layout: AutoGazeTextureLayout,
}

impl Default for AutoGazeTexture {
    fn default() -> Self {
        Self {
            image: Handle::default(),
            input_image: Handle::default(),
            mask_image: Handle::default(),
            output_image: Handle::default(),
            entity: None,
            side_by_side_entity: None,
            input_entity: None,
            mask_entity: None,
            output_entity: None,
            width: 3,
            height: 1,
            layout: AutoGazeTextureLayout::default(),
        }
    }
}

#[derive(Component)]
pub(crate) struct OneShotGpuUpload;

pub(crate) enum VisualizationImageData {
    SideBySideRgba(Vec<u8>),
    PanelsRgba {
        panel_width: u32,
        panel_height: u32,
        input_rgba: Vec<u8>,
        mask_rgba: Vec<u8>,
        output_rgba: Vec<u8>,
    },
    TensorPanels(Box<TensorPanelVisualizationData>),
}

pub(crate) struct TensorPanelVisualizationData {
    pub(crate) panel_width: u32,
    pub(crate) panel_height: u32,
    pub(crate) input_rgba: Tensor<AutoGazeBevyBackend, 3>,
    pub(crate) mask_rgba: Tensor<AutoGazeBevyBackend, 3>,
    pub(crate) output_rgba: Tensor<AutoGazeBevyBackend, 3>,
}

pub(crate) struct Visualization {
    pub(crate) width: u32,
    pub(crate) height: u32,
    #[cfg(test)]
    pub(crate) rgba: Vec<u8>,
    #[cfg(test)]
    pub(crate) tensor: Option<Tensor<AutoGazeBevyBackend, 3>>,
    pub(crate) image_data: VisualizationImageData,
    pub(crate) gaze_update_ratio: f64,
    pub(crate) interframe_keyframe: bool,
    pub(crate) psnr_db: Option<f64>,
    pub(crate) visualize_cpu_ms: f64,
    pub(crate) psnr_ms: f64,
    pub(crate) tensor_ms: f64,
    pub(crate) output_rgba_bytes: usize,
    pub(crate) output_tensor_bytes: usize,
    pub(crate) tensor_interframe_path: Option<AutoGazeTensorInterframePath>,
    pub(crate) effective_display_transfer: BevyDisplayTransfer,
    pub(crate) mask_plan_stats: AutoGazeMaskPlanStats,
    pub(crate) timing: Option<crate::InferenceTiming>,
}

pub(crate) fn apply_visualization_to_texture(
    visualization: Visualization,
    texture: &mut AutoGazeTexture,
    images: &mut Assets<Image>,
) {
    let width = visualization.width;
    let height = visualization.height;
    match visualization.image_data {
        VisualizationImageData::SideBySideRgba(rgba) => {
            set_visualization_image(&texture.image, width, height, rgba, images);
            texture.layout = AutoGazeTextureLayout::SideBySide;
        }
        VisualizationImageData::PanelsRgba {
            panel_width,
            panel_height,
            input_rgba,
            mask_rgba,
            output_rgba,
        } => {
            set_panel_visualization_images(
                texture,
                images,
                PanelVisualizationImages {
                    width: panel_width,
                    height: panel_height,
                    input_rgba,
                    mask_rgba,
                    output_rgba,
                },
            );
            texture.layout = AutoGazeTextureLayout::Panels;
        }
        VisualizationImageData::TensorPanels(_) => {}
    }
    texture.width = width;
    texture.height = height;
}

pub(crate) fn apply_visualization_to_world(
    world: &mut World,
    width: u32,
    height: u32,
    image_data: VisualizationImageData,
) {
    let Some(texture) = world.get_resource::<AutoGazeTexture>().cloned() else {
        return;
    };

    match image_data {
        VisualizationImageData::TensorPanels(panels) => {
            let TensorPanelVisualizationData {
                panel_width,
                panel_height,
                input_rgba,
                mask_rgba,
                output_rgba,
            } = *panels;
            set_texture_layout(world, &texture, AutoGazeTextureLayout::Panels);
            remove_gpu_visualization_handle(world, texture.side_by_side_entity);
            if let Some(mut images) = world.get_resource_mut::<Assets<Image>>() {
                set_gpu_visualization_image(
                    &texture.input_image,
                    panel_width,
                    panel_height,
                    &mut images,
                );
                set_gpu_visualization_image(
                    &texture.mask_image,
                    panel_width,
                    panel_height,
                    &mut images,
                );
                set_gpu_visualization_image(
                    &texture.output_image,
                    panel_width,
                    panel_height,
                    &mut images,
                );
            }
            set_gpu_panel_upload_handle(
                world,
                texture.input_entity,
                texture.input_image.clone(),
                input_rgba,
            );
            set_gpu_panel_upload_handle(
                world,
                texture.mask_entity,
                texture.mask_image.clone(),
                mask_rgba,
            );
            set_gpu_panel_upload_handle(
                world,
                texture.output_entity,
                texture.output_image.clone(),
                output_rgba,
            );
        }
        VisualizationImageData::SideBySideRgba(rgba) => {
            set_texture_layout(world, &texture, AutoGazeTextureLayout::SideBySide);
            remove_panel_gpu_visualization_handles(world, &texture);
            remove_gpu_visualization_handle(world, texture.side_by_side_entity);
            if let Some(mut images) = world.get_resource_mut::<Assets<Image>>() {
                set_visualization_image(&texture.image, width, height, rgba, &mut images);
            }
        }
        VisualizationImageData::PanelsRgba {
            panel_width,
            panel_height,
            input_rgba,
            mask_rgba,
            output_rgba,
        } => {
            set_texture_layout(world, &texture, AutoGazeTextureLayout::Panels);
            remove_gpu_visualization_handle(world, texture.side_by_side_entity);
            remove_panel_gpu_visualization_handles(world, &texture);
            if let Some(mut images) = world.get_resource_mut::<Assets<Image>>() {
                set_panel_visualization_images(
                    &texture,
                    &mut images,
                    PanelVisualizationImages {
                        width: panel_width,
                        height: panel_height,
                        input_rgba,
                        mask_rgba,
                        output_rgba,
                    },
                );
            }
        }
    }
}

pub(crate) fn visualization_image(width: u32, height: u32, mut rgba: Vec<u8>) -> Image {
    let width = width.max(1);
    let height = height.max(1);
    let expected_len = width as usize * height as usize * 4;
    if rgba.len() != expected_len {
        rgba.resize(expected_len, 0);
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
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    image.sampler = ImageSampler::nearest();
    image
}

fn set_texture_layout(world: &mut World, texture: &AutoGazeTexture, layout: AutoGazeTextureLayout) {
    if let Some(entity) = texture.side_by_side_entity {
        set_node_display(
            world,
            entity,
            if layout == AutoGazeTextureLayout::SideBySide {
                Display::Flex
            } else {
                Display::None
            },
        );
    }

    for entity in [
        texture.input_entity,
        texture.mask_entity,
        texture.output_entity,
    ]
    .into_iter()
    .flatten()
    {
        set_node_display(
            world,
            entity,
            if layout == AutoGazeTextureLayout::Panels {
                Display::Flex
            } else {
                Display::None
            },
        );
    }

    if let Some(mut texture) = world.get_resource_mut::<AutoGazeTexture>() {
        texture.layout = layout;
    }
}

fn set_node_display(world: &mut World, entity: Entity, display: Display) {
    if let Ok(mut entity) = world.get_entity_mut(entity)
        && let Some(mut node) = entity.get_mut::<Node>()
    {
        node.display = display;
    }
}

fn remove_gpu_visualization_handle(world: &mut World, entity: Option<Entity>) {
    if let Some(entity) = entity
        && let Ok(mut entity) = world.get_entity_mut(entity)
    {
        entity.remove::<BevyBurnHandle<AutoGazeBevyBackend>>();
    }
}

fn remove_panel_gpu_visualization_handles(world: &mut World, texture: &AutoGazeTexture) {
    remove_gpu_visualization_handle(world, texture.input_entity);
    remove_gpu_visualization_handle(world, texture.mask_entity);
    remove_gpu_visualization_handle(world, texture.output_entity);
}

fn set_gpu_panel_upload_handle(
    world: &mut World,
    entity: Option<Entity>,
    image: Handle<Image>,
    tensor: Tensor<AutoGazeBevyBackend, 3>,
) {
    let Some(entity) = entity else {
        return;
    };
    let Ok(mut entity) = world.get_entity_mut(entity) else {
        return;
    };
    if let Some(mut handle) = entity.get_mut::<BevyBurnHandle<AutoGazeBevyBackend>>() {
        handle.bevy_image = image;
        handle.tensor = tensor;
        handle.direction = BindingDirection::BurnToBevy;
        handle.xfer = TransferKind::Gpu;
        handle.upload = true;
    } else {
        entity.insert(BevyBurnHandle::<AutoGazeBevyBackend> {
            bevy_image: image,
            tensor,
            upload: true,
            direction: BindingDirection::BurnToBevy,
            xfer: TransferKind::Gpu,
        });
    }
    entity.insert(OneShotGpuUpload);
}

struct PanelVisualizationImages {
    width: u32,
    height: u32,
    input_rgba: Vec<u8>,
    mask_rgba: Vec<u8>,
    output_rgba: Vec<u8>,
}

fn set_panel_visualization_images(
    texture: &AutoGazeTexture,
    images: &mut Assets<Image>,
    panels: PanelVisualizationImages,
) {
    let PanelVisualizationImages {
        width,
        height,
        input_rgba,
        mask_rgba,
        output_rgba,
    } = panels;
    set_visualization_image(&texture.input_image, width, height, input_rgba, images);
    set_visualization_image(&texture.mask_image, width, height, mask_rgba, images);
    set_visualization_image(&texture.output_image, width, height, output_rgba, images);
}

fn set_visualization_image(
    handle: &Handle<Image>,
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    images: &mut Assets<Image>,
) {
    if let Some(mut image) = images.get_mut(handle)
        && image.width() == width
        && image.height() == height
        && image.texture_descriptor.format == TextureFormat::Rgba8UnormSrgb
        && image
            .texture_descriptor
            .usage
            .contains(TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING)
    {
        image.data = Some(rgba);
        return;
    }

    let _ = images.insert(handle.id(), visualization_image(width, height, rgba));
}

fn set_gpu_visualization_image(
    handle: &Handle<Image>,
    width: u32,
    height: u32,
    images: &mut Assets<Image>,
) {
    let width = width.max(1);
    let height = height.max(1);
    if let Some(image) = images.get(handle)
        && image.width() == width
        && image.height() == height
        && image.texture_descriptor.format == TextureFormat::Rgba32Float
        && image.texture_descriptor.usage.contains(
            TextureUsages::COPY_DST
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING,
        )
    {
        return;
    }

    let _ = images.insert(handle.id(), gpu_visualization_image(width, height));
}

fn gpu_visualization_image(width: u32, height: u32) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::STORAGE_BINDING;
    image.sampler = ImageSampler::nearest();
    image
}

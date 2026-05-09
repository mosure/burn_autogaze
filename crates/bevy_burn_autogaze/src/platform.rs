#[cfg(feature = "native")]
pub mod camera {
    use std::sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, Sender, SyncSender, TryRecvError},
    };
    use std::time::Duration;

    use image::{RgbImage, RgbaImage};
    use nokhwa::{
        Camera, nokhwa_initialize,
        pixel_format::RgbFormat,
        query,
        utils::{
            ApiBackend, CameraFormat, CameraInfo, FrameFormat, RequestedFormat, RequestedFormatType,
        },
    };
    use once_cell::sync::OnceCell;

    const CAMERA_WIDTH: u32 = 1280;
    const CAMERA_HEIGHT: u32 = 720;
    const CAMERA_FPS: u32 = 30;

    #[derive(Clone, Copy, Debug)]
    pub struct CameraRequest {
        pub width: u32,
        pub height: u32,
        pub fps: u32,
    }

    impl CameraRequest {
        pub const fn new(width: u32, height: u32, fps: u32) -> Self {
            Self { width, height, fps }
        }
    }

    impl Default for CameraRequest {
        fn default() -> Self {
            Self::new(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
        }
    }

    pub static SAMPLE_RECEIVER: OnceCell<Arc<Mutex<Receiver<RgbaImage>>>> = OnceCell::new();
    pub static SAMPLE_SENDER: OnceCell<SyncSender<RgbaImage>> = OnceCell::new();

    pub static APP_RUN_RECEIVER: OnceCell<Arc<Mutex<Receiver<()>>>> = OnceCell::new();
    pub static APP_RUN_SENDER: OnceCell<Sender<()>> = OnceCell::new();

    pub fn native_camera_thread() {
        native_camera_thread_with_request(CameraRequest::default());
    }

    pub fn native_camera_thread_with_request(request: CameraRequest) {
        let (sample_sender, sample_receiver) = mpsc::sync_channel(1);
        if SAMPLE_RECEIVER
            .set(Arc::new(Mutex::new(sample_receiver)))
            .is_err()
        {
            crate::log("camera sample receiver already initialized");
            return;
        }
        if SAMPLE_SENDER.set(sample_sender).is_err() {
            crate::log("camera sample sender already initialized");
            return;
        }

        let (app_run_sender, app_run_receiver) = mpsc::channel();
        if APP_RUN_RECEIVER
            .set(Arc::new(Mutex::new(app_run_receiver)))
            .is_err()
        {
            crate::log("camera stop receiver already initialized");
            return;
        }
        if APP_RUN_SENDER.set(app_run_sender).is_err() {
            crate::log("camera stop sender already initialized");
            return;
        }

        nokhwa_initialize(|granted| {
            if !granted {
                crate::log("camera permission was not granted");
            }
        });

        crate::log("querying native cameras...");
        let devices = match query(ApiBackend::Auto) {
            Ok(devices) => devices,
            Err(err) => {
                crate::log(&format!("failed to query cameras: {err}"));
                return;
            }
        };
        crate::log(&format!("found {} native camera(s)", devices.len()));
        if devices.is_empty() {
            crate::log("no camera found");
            return;
        }

        let Some(mut camera) = open_first_camera(&devices, request) else {
            crate::log("failed to open any native camera");
            return;
        };

        let mut error_count = 0usize;

        loop {
            if should_stop() {
                break;
            }

            match camera.frame() {
                Ok(buffer) => match buffer.decode_image::<RgbFormat>() {
                    Ok(image) => {
                        error_count = 0;
                        if let Some(sender) = SAMPLE_SENDER.get() {
                            let _ = sender.try_send(rgb_to_rgba(image));
                        }
                    }
                    Err(err) => {
                        error_count += 1;
                        log_capture_error("decode camera frame", err, error_count);
                    }
                },
                Err(err) => {
                    error_count += 1;
                    log_capture_error("capture camera frame", err, error_count);
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }

        if let Err(err) = camera.stop_stream() {
            crate::log(&format!("failed to stop camera stream: {err}"));
        }
        crate::log("camera stream stopped");
    }

    pub fn receive_image() -> Option<RgbaImage> {
        let receiver = SAMPLE_RECEIVER.get()?;
        let mut last_image = None;

        {
            let Ok(receiver) = receiver.lock() else {
                return None;
            };
            while let Ok(image) = receiver.try_recv() {
                last_image = Some(image);
            }
        }

        last_image
    }

    fn rgb_to_rgba(image: RgbImage) -> RgbaImage {
        let (width, height) = image.dimensions();
        let rgb = image.into_raw();
        let mut rgba = vec![255u8; width as usize * height as usize * 4];
        for (src, dst) in rgb.chunks_exact(3).zip(rgba.chunks_exact_mut(4)) {
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
        }
        RgbaImage::from_raw(width, height, rgba).expect("valid rgba frame")
    }

    fn open_first_camera(devices: &[CameraInfo], request: CameraRequest) -> Option<Camera> {
        let requested_formats = [
            (
                "closest requested MJPEG",
                RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(
                    CameraFormat::new_from(
                        request.width.max(1),
                        request.height.max(1),
                        FrameFormat::MJPEG,
                        request.fps.max(1),
                    ),
                )),
            ),
            (
                "closest requested YUYV",
                RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(
                    CameraFormat::new_from(
                        request.width.max(1),
                        request.height.max(1),
                        FrameFormat::YUYV,
                        request.fps.max(1),
                    ),
                )),
            ),
            (
                "backend default",
                RequestedFormat::new::<RgbFormat>(RequestedFormatType::None),
            ),
        ];

        for camera_info in devices {
            for (label, requested_format) in requested_formats {
                let index = camera_info.index().clone();
                crate::log(&format!(
                    "opening camera `{}` ({index}) with {label} near {}x{}@{}",
                    camera_info.human_name(),
                    request.width.max(1),
                    request.height.max(1),
                    request.fps.max(1),
                ));
                let mut camera = match Camera::new(index, requested_format) {
                    Ok(camera) => camera,
                    Err(err) => {
                        crate::log(&format!("failed to create camera with {label}: {err}"));
                        continue;
                    }
                };

                let format = camera.camera_format();
                crate::log(&format!(
                    "camera format: {}x{} {}fps {}",
                    format.width(),
                    format.height(),
                    format.frame_rate(),
                    format.format()
                ));

                match camera.open_stream() {
                    Ok(()) => {
                        crate::log("camera stream open");
                        return Some(camera);
                    }
                    Err(err) => {
                        crate::log(&format!("failed to open camera stream with {label}: {err}"));
                    }
                }
            }
        }

        None
    }

    fn should_stop() -> bool {
        let Some(receiver) = APP_RUN_RECEIVER.get() else {
            return true;
        };
        let Ok(receiver) = receiver.lock() else {
            crate::log("camera stop receiver was poisoned");
            return true;
        };
        match receiver.try_recv() {
            Ok(_) | Err(TryRecvError::Disconnected) => true,
            Err(TryRecvError::Empty) => false,
        }
    }

    fn log_capture_error(action: &str, err: impl std::fmt::Display, error_count: usize) {
        if error_count == 1 || error_count.is_multiple_of(120) {
            crate::log(&format!("failed to {action}: {err}"));
        }
    }
}

#[cfg(feature = "web")]
pub mod camera {
    use std::cell::RefCell;

    use image::RgbaImage;
    use wasm_bindgen::prelude::*;

    thread_local! {
        pub static SAMPLE_RECEIVER: RefCell<Option<RgbaImage>> = const { RefCell::new(None) };
    }

    #[wasm_bindgen]
    pub fn frame_input(pixel_data: &[u8], width: u32, height: u32) {
        let Some(image) = RgbaImage::from_raw(width, height, pixel_data.to_vec()) else {
            crate::log("ignoring invalid RGBA frame input");
            return;
        };
        SAMPLE_RECEIVER.with(|receiver| {
            *receiver.borrow_mut() = Some(image);
        });
    }

    pub fn receive_image() -> Option<RgbaImage> {
        SAMPLE_RECEIVER.with(|receiver| receiver.borrow_mut().take())
    }
}

#[cfg(feature = "native")]
pub mod camera {
    use std::sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, Sender, SyncSender, TryRecvError},
    };

    use image::{RgbImage, RgbaImage};
    use nokhwa::{
        CallbackCamera, nokhwa_initialize,
        pixel_format::RgbFormat,
        query,
        utils::{ApiBackend, RequestedFormat, RequestedFormatType},
    };
    use once_cell::sync::OnceCell;

    pub static SAMPLE_RECEIVER: OnceCell<Arc<Mutex<Receiver<RgbaImage>>>> = OnceCell::new();
    pub static SAMPLE_SENDER: OnceCell<SyncSender<RgbaImage>> = OnceCell::new();

    pub static APP_RUN_RECEIVER: OnceCell<Arc<Mutex<Receiver<()>>>> = OnceCell::new();
    pub static APP_RUN_SENDER: OnceCell<Sender<()>> = OnceCell::new();

    pub fn native_camera_thread() {
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

        let devices = match query(ApiBackend::Auto) {
            Ok(devices) => devices,
            Err(err) => {
                crate::log(&format!("failed to query cameras: {err}"));
                return;
            }
        };
        let Some(index) = devices.first().map(|device| device.index()) else {
            crate::log("no camera found");
            return;
        };

        let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);
        let camera = CallbackCamera::new(index.clone(), format, |buffer| {
            let Ok(image) = buffer.decode_image::<RgbFormat>() else {
                return;
            };
            if let Some(sender) = SAMPLE_SENDER.get() {
                let _ = sender.try_send(rgb_to_rgba(image));
            }
        });
        let Ok(mut camera) = camera else {
            crate::log("failed to open camera");
            return;
        };

        if let Err(err) = camera.open_stream() {
            crate::log(&format!("failed to open camera stream: {err}"));
            return;
        }

        loop {
            if let Err(err) = camera.poll_frame() {
                crate::log(&format!("failed to poll camera frame: {err}"));
                break;
            }

            let Some(receiver) = APP_RUN_RECEIVER.get() else {
                break;
            };
            let Ok(receiver) = receiver.lock() else {
                crate::log("camera stop receiver was poisoned");
                break;
            };
            match receiver.try_recv() {
                Ok(_) => break,
                Err(TryRecvError::Empty) => continue,
                Err(TryRecvError::Disconnected) => break,
            };
        }

        if let Err(err) = camera.stop_stream() {
            crate::log(&format!("failed to stop camera stream: {err}"));
        }
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
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
        for pixel in image.into_raw().chunks_exact(3) {
            rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
        }
        RgbaImage::from_raw(width, height, rgba).expect("valid rgba frame")
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

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
        SAMPLE_RECEIVER
            .set(Arc::new(Mutex::new(sample_receiver)))
            .unwrap();
        SAMPLE_SENDER.set(sample_sender).unwrap();

        let (app_run_sender, app_run_receiver) = mpsc::channel();
        APP_RUN_RECEIVER
            .set(Arc::new(Mutex::new(app_run_receiver)))
            .unwrap();
        APP_RUN_SENDER.set(app_run_sender).unwrap();

        nokhwa_initialize(|granted| {
            if !granted {
                panic!("failed to initialize camera");
            }
        });

        let devices = query(ApiBackend::Auto).expect("failed to query cameras");
        let index = devices.first().expect("no camera found").index();

        let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);
        let mut camera = CallbackCamera::new(index.clone(), format, |buffer| {
            let image = buffer.decode_image::<RgbFormat>().unwrap();
            let sender = SAMPLE_SENDER.get().unwrap();
            sender.send(rgb_to_rgba(image)).unwrap();
        })
        .expect("failed to open camera");

        camera.open_stream().expect("failed to open camera stream");

        loop {
            camera.poll_frame().expect("failed to poll camera frame");

            let receiver = APP_RUN_RECEIVER.get().unwrap();
            match receiver.lock().unwrap().try_recv() {
                Ok(_) => break,
                Err(TryRecvError::Empty) => continue,
                Err(TryRecvError::Disconnected) => break,
            };
        }

        camera.stop_stream().expect("failed to stop camera stream");
    }

    pub fn receive_image() -> Option<RgbaImage> {
        let receiver = SAMPLE_RECEIVER.get()?;
        let mut last_image = None;

        {
            let receiver = receiver.lock().unwrap();
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
        let image = RgbaImage::from_raw(width, height, pixel_data.to_vec())
            .expect("failed to create RGBA frame");
        SAMPLE_RECEIVER.with(|receiver| {
            *receiver.borrow_mut() = Some(image);
        });
    }

    pub fn receive_image() -> Option<RgbaImage> {
        SAMPLE_RECEIVER.with(|receiver| receiver.borrow_mut().take())
    }
}

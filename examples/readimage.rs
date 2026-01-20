use eframe::egui;
use egui_extras::RetainedImage;
use image::GrayImage;

struct ImageViewer {
    image: Option<RetainedImage>,
}

impl ImageViewer {
    fn new() -> Self {
        // 加载图像并转换为egui兼容格式
        if let Ok(img) = image::open("image.png") {
            let img = img.to_luma8(); // 转换为灰度图像
            let (width, height) = img.dimensions();

            // 将灰度图像转换为RGBA格式以便显示
            let mut rgba_data = Vec::with_capacity((width * height) as usize * 4);
            for y in 0..height {
                for x in 0..width {
                    let pixel = img.get_pixel(x, y)[0];
                    rgba_data.push(pixel); // R
                    rgba_data.push(pixel); // G
                    rgba_data.push(pixel); // B
                    rgba_data.push(255);   // A
                }
            }

            // 创建 egui 兼容的图像
            let color_image = egui::ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &rgba_data);
            let image = RetainedImage::from_color_image("loaded_image", color_image);

            Self {
                image: Some(image),
            }
        } else {
            Self {
                image: None,
            }
        }
    }
}

impl eframe::App for ImageViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Image Viewer");

            if let Some(img) = &self.image {
                // 自动调整图片大小以适应可用空间
                ui.image((img.texture_id(ctx), img.size_vec2()));
            } else {
                ui.label("Failed to load image");
            }
        });

        ctx.request_repaint();
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::Vec2::new(800.0, 600.0)),
        ..Default::default()
    };

    eframe::run_native(
        "Image Viewer",
        options,
        Box::new(|_cc| Ok(Box::new(ImageViewer::new()))),
    )
}
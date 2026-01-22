use anyhow::{Result, Context};
use memmap2::Mmap;
use std::fs::File;
use std::io::{Cursor};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use ort::session::Session;
use ndarray::Array3;

#[derive(Debug, Clone)]
pub enum ImageFormat {
    Png,
    Jpeg,
}

pub struct StreamingImageProcessor {
    format: ImageFormat,
    width: u32,
    height: u32,
    channels: u8,
    file_data: Arc<Mmap>,
}

impl StreamingImageProcessor {
    /// 从文件创建流式处理器（不加载完整图片到内存）
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let start = Instant::now();
        let file = File::open(&path)
            .with_context(|| format!("无法打开文件: {}", path.as_ref().display()))?;

        let mmap = unsafe { Mmap::map(&file)? };
        let format = detect_format(&mmap)?;
        let mmap_size = mmap.len();
        println!("全量 mmap 大小：{} 字节（{:.2} MB）", mmap_size, mmap_size as f64 / 1024.0 / 1024.0);

        // 只读取图片头信息，不加载像素数据
        let (width, height, channels) = read_image_metadata(&mmap, &format)?;

        println!("检测到图片: {}x{}, {} 通道, 格式: {:?}",
                width, height, channels, format);
        println!("预估内存需求: {:.2} MB",
                (width as u64 * height as u64 * channels as u64) as f64 / 1024.0 / 1024.0);
        let end = start.elapsed();
        println!("初始化processor:: {}us", end.as_micros());
        Ok(Self {
            format,
            width,
            height,
            channels,
            file_data: Arc::new(mmap),
        })
    }

    /// 获取图片基本信息
    pub fn info(&self) -> (u32, u32, u8) {
        (self.width, self.height, self.channels)
    }

    /// 流式读取指定区域的像素数据
    pub fn read_region(&self, decoded_buffer: &[u8], x: u32, y: u32, width: u32, height: u32) -> Result<Vec<u8>> {
        if x + width > self.width || y + height > self.height {
            anyhow::bail!("读取区域超出图片边界");
        }

        match self.format {
            ImageFormat::Png => self.read_png_region(x, y, width, height),
            ImageFormat::Jpeg => self.read_jpeg_region(decoded_buffer, x, y, width, height),
        }
    }

    /// PNG 流式区域读取
    fn read_png_region(&self, x: u32, y: u32, width: u32, height: u32) -> Result<Vec<u8>> {
        let cursor = Cursor::new(&*self.file_data);
        let decoder = png::Decoder::new(cursor);
        let mut reader = decoder.read_info()?;
        let buffer_size = reader.output_buffer_size().unwrap();
        let mut row_buffer = vec![0; buffer_size];
        let bytes_per_pixel = self.channels as usize;
        let _bytes_per_row = self.width as usize * bytes_per_pixel;

        let mut result = Vec::with_capacity((width * height * self.channels as u32) as usize);

        // 逐行读取，只保留需要的区域
        for row_idx in 0..self.height {
            reader.read_row(&mut row_buffer)?;

            if row_idx >= y && row_idx < y + height {
                let start_col = (x as usize) * bytes_per_pixel;
                let end_col = ((x + width) as usize) * bytes_per_pixel;
                result.extend_from_slice(&row_buffer[start_col..end_col]);
            }
        }

        Ok(result)
    }

    fn decode_jpg(&self) -> Result<Vec<u8>> {
        let cursor = Cursor::new(&*self.file_data);
        let mut decoder = jpeg_decoder::Decoder::new(cursor);
        decoder.read_info()?;
        let info = decoder.info().ok_or("无法获取info").unwrap();
        println!("JPEG pixel format: {:?} 大小 {}*{}", info.pixel_format, info.width, info.height);
        let start = Instant::now();
        println!("jpeg 开始解码:::---");
        // 转灰度图
        // decoder.set_color_transform(jpeg_decoder::ColorTransform::Grayscale);
        let pixels = decoder.decode()?;

        let mut gray_pixels = Vec::with_capacity((info.width as usize) * (info.height as usize));
        for chunk in pixels.chunks(3) {
            let r = chunk[0] as f32;
            let g = chunk[1] as f32;
            let b = chunk[2] as f32;
            let gray = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
            gray_pixels.push(gray);
        }
        let end = start.elapsed();
        println!("jpeg 解压并转灰度后大小：{}MB， jpeg 解码耗时：：：{}ms", gray_pixels.len()/1024/1024, end.as_millis());
        drop(pixels);
        Ok(gray_pixels)
    }

    /// JPEG 流式区域读取（JPEG不支持完全的流式读取，但可以优化内存使用）
    fn read_jpeg_region(&self, decoded_buffer: &[u8], x: u32, y: u32, width: u32, height: u32) -> Result<Vec<u8>> {
        // JPEG 需要解压整个图片，但我们可以立即丢弃不需要的行
        let start = Instant::now();
        let bytes_per_pixel = self.channels as usize;
        let mut result = Vec::with_capacity((width * height * self.channels as u32) as usize);
        // 从完整像素数据中提取需要的区域
        for row in y..(y + height) {
            let row_start = (row * self.width) as usize * bytes_per_pixel;
            let start_idx = row_start + (x as usize * bytes_per_pixel);
            let end_idx = start_idx + (width as usize * bytes_per_pixel);
            result.extend_from_slice(&decoded_buffer[start_idx..end_idx]);
        }
        let end = start.elapsed();
        println!("读取特定区域耗时：：：{}", end.as_millis());
        Ok(result)
    }
}

/// 检测图片格式
fn detect_format(data: &[u8]) -> Result<ImageFormat> {
    if data.len() < 8 {
        anyhow::bail!("文件太小，无法检测格式");
    }

    // PNG signature: 89 50 4E 47 0D 0A 1A 0A
    if data.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Ok(ImageFormat::Png);
    }

    // JPEG signature: FF D8 FF
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Ok(ImageFormat::Jpeg);
    }

    anyhow::bail!("不支持的图片格式");
}

/// 只读取图片元数据，不加载像素数据
fn read_image_metadata(data: &[u8], format: &ImageFormat) -> Result<(u32, u32, u8)> {
    match format {
        ImageFormat::Png => {
            let cursor = Cursor::new(data);
            let decoder = png::Decoder::new(cursor);
            let reader = decoder.read_info()?;
            let info = reader.info();
            let channels = match info.color_type {
                png::ColorType::Grayscale => 1,
                png::ColorType::Rgb => 3,
                png::ColorType::Rgba => 4,
                png::ColorType::GrayscaleAlpha => 2,
                png::ColorType::Indexed => 1,
            };
            Ok((info.width, info.height, channels))
        },
        ImageFormat::Jpeg => {
            let cursor = Cursor::new(data);
            let mut decoder = jpeg_decoder::Decoder::new(cursor);
            let _metadata = decoder.read_info()?;
            let (width, height, channels) = match decoder.info() {
                Some(info) => (info.width as u32, info.height as u32, info.pixel_format.pixel_bytes() as u8),
                None => (0, 0, 3),
            };
            println!("====================channels:{}", channels);
            Ok((width, height, channels))
        },
    }
}

// 将region_data转换为ONNX模型的输入张量
fn convert_region_to_tensor(region_data: &[u8], height: u32, width: u32, channels: u8) -> Result<Array3<f32>> {
    // 将u8数据转换为f32，并归一化到[0,1]范围
    let normalized_data: Vec<f32> = region_data
        .iter()
        .map(|&x| x as f32 / 255.0)
        .collect();

    // 根据通道数重塑数组
    // 如果是单通道图像，需要扩展维度
    if channels == 1 {
        // 单通道图像: [H, W] -> [1, H, W]
        let tensor = Array3::from_shape_vec((1, height as usize, width as usize), normalized_data)
            .map_err(|_| anyhow::anyhow!("无法重塑数组形状"))?;
        Ok(tensor)
    } else if channels == 3 {
        // RGB图像: [H, W, C] -> [C, H, W] (CHW格式)
        let mut reshaped = Array3::zeros((channels as usize, height as usize, width as usize));
        for (i, &pixel_value) in normalized_data.iter().enumerate() {
            let c = (i % 3) as usize;  // 通道索引
            let idx_in_channel = i / 3;
            let h = idx_in_channel / (width as usize);
            let w = idx_in_channel % (width as usize);
            reshaped[[c, h, w]] = pixel_value;
        }
        Ok(reshaped)
    } else {
        // 其他通道数的情况
        anyhow::bail!("不支持的通道数: {}", channels);
    }
}

fn main() -> Result<()> {
    // 创建流式处理器
    let processor = StreamingImageProcessor::from_file("big.jpg")?;

    let (width, height, channels) = processor.info();
    println!("图片信息: {}x{}, {} 通道", width, height, channels);

    let decoded_buffer = processor.decode_jpg()?;

    // 加载模型
    let model_path = "unet.onnx";
    let session = Session::builder()?
        .edit_from_file(model_path)?;

    println!("模型加载成功！");
    println!("模型输入数量: {}", session.inputs().len());
    println!("模型输出数量: {}", session.outputs().len());

    // 获取模型输入信息
    let input_info = &session.inputs()[0];
    println!("输入名称: {}", input_info.name());

    // 读取一个区域作为示例
    let region_width = 1024;
    let region_height = 1024;
    let region_data = processor.read_region(&decoded_buffer, 0, 0, region_width, region_height)?;

    println!("读取区域数据大小: {} bytes", region_data.len());

    // 将region_data转换为ONNX模型的输入张量
    // 注意：由于我们从JPEG解码后得到了灰度图（1通道），所以这里使用1作为通道数
    let tensor = convert_region_to_tensor(&region_data, region_height, region_width, 1)?;
    println!("张量形状: {:?}", tensor.shape());

    // 现在我们展示了如何将region_data转换为张量
    // 实际的推理部分需要根据具体模型的输入要求进行调整
    println!("region_data 已成功转换为张量格式，准备用于ONNX模型推理");
    println!("转换步骤:");
    println!("1. 读取图像区域数据");
    println!("2. 将u8数据转换为f32并归一化到[0,1]");
    println!("3. 重塑数据为正确的形状 (CHW格式)");
    println!("4. 使用ORT API将ndarray转换为ORT张量");

    Ok(())
}
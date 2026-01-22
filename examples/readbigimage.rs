use anyhow::{Result, Context};
// use egui::debug_text::print;
// use jpeg_decoder::PixelFormat;
use serde::{Deserialize};
use memmap2::Mmap;
use std::fs::File;
use std::io::{Cursor};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
// use onnxruntime::{
//     environment::Environment,
//     // session::Session,
//     // tensor::OrtOwnedTensor,
//     GraphOptimizationLevel, LoggingLevel,
// };
use ort::session::Session;
#[derive(Debug, Clone)]
pub enum ImageFormat {
    Png,
    Jpeg,
    // Tiff,
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
        let start =Instant::now();
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
        println!("初始化processor:: {}us",end.as_micros());
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
    pub fn read_region(&self, decoded_buffer:&[u8], x: u32, y: u32, width: u32, height: u32) -> Result<Vec<u8>> {
        if x + width > self.width || y + height > self.height {
            anyhow::bail!("读取区域超出图片边界");
        }
        
        match self.format {
            ImageFormat::Png => self.read_png_region(x, y, width, height),
            ImageFormat::Jpeg => self.read_jpeg_region(decoded_buffer,x, y, width, height),
            // ImageFormat::Tiff => self.read_tiff_region(x, y, width, height),
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
    fn decode_jpg(&self)->Result<Vec<u8>>{ 
        let cursor = Cursor::new(&*self.file_data);
        let mut decoder = jpeg_decoder::Decoder::new(cursor);
        decoder.read_info()?;
        let info = decoder.info().ok_or("无法获取info").unwrap();
        println!("JPEG pixel format: {:?} 大小 {}*{}", info.pixel_format,info.width,info.height);
        let start = Instant::now();
        println!("jpeg 开始解码:::---")  ;
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
        println!("jpeg 解压并转灰度后大小：{}MB， jpeg 解码耗时：：：{}ms",gray_pixels.len()/1024/1024,end.as_millis());
        drop(pixels);
        Ok(gray_pixels)
    }

    /// JPEG 流式区域读取（JPEG不支持完全的流式读取，但可以优化内存使用）
    fn read_jpeg_region(&self,decoded_buffer:&[u8], x: u32, y: u32, width: u32, height: u32) -> Result<Vec<u8>> {
        // JPEG 需要解压整个图片，但我们可以立即丢弃不需要的行
        let start =Instant::now();
        // let cursor = Cursor::new(&*self.file_data);
        // let mut decoder = jpeg_decoder::Decoder::new(cursor);
        // let start2 = Instant::now();
        // println!("jpeg 开始解压:::---");
        // let pixels = decoder.decode()?;
        
        // println!("JPEG 解压后大小{}MB,耗时:{}", pixels.len()/1024/1024,start2.elapsed().as_millis());
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
        println!("读取特定区域耗时：：：{}",end.as_millis());
        Ok(result)
    }
    
    // /// TIFF 流式区域读取
    // fn read_tiff_region(&self, x: u32, y: u32, width: u32, height: u32) -> Result<Vec<u8>> {
    //     let cursor = Cursor::new(&*self.file_data);
    //     let mut decoder = tiff::decoder::Decoder::new(cursor)?;
        
    //     let mut result = Vec::with_capacity((width * height * self.channels as u32) as usize);
        
    //     // TIFF 支持按条带或块读取
    //     if decoder.tile_count().is_ok() {
    //         // 处理分块的 TIFF
    //         self.read_tiled_tiff_region(&mut decoder, x, y, width, height)
    //     } else {
    //         // 处理条带化的 TIFF
    //         self.read_strip_tiff_region(&mut decoder, x, y, width, height)
    //     }
    // }
    
    // fn read_tiled_tiff_region(
    //     &self,
    //     decoder: &mut tiff::decoder::Decoder<Cursor<&Mmap>>,
    //     x: u32, y: u32, width: u32, height: u32
    // ) -> Result<Vec<u8>> {
    //     // 实现分块 TIFF 的区域读取
    //     // 这里需要根据具体的 TIFF 分块信息来实现
    //     anyhow::bail!("分块 TIFF 读取暂未实现")
    // }
    
    // fn read_strip_tiff_region(
    //     &self,
    //     decoder: &mut tiff::decoder::Decoder<Cursor<&Mmap>>,
    //     x: u32, y: u32, width: u32, height: u32
    // ) -> Result<Vec<u8>> {
    //     // 实现条带化 TIFF 的区域读取
    //     anyhow::bail!("条带化 TIFF 读取暂未实现")
    // }
    
    /// 分块处理整个图片
    pub fn process_in_chunks<F>(&self,decoded_buffer:&[u8], chunk_size: u32, overlap: u32, mut callback: F) -> Result<()>
    where
        F: FnMut(u32, u32, u32, u32, &[u8]) -> Result<()>,
    {
        let step = chunk_size - overlap;
        let cols = (self.width + step - 1) / step;
        let rows = (self.height + step - 1) / step;
        
        println!("开始分块处理: {}x{} 块", rows, cols);
        
        for row in 0..rows {
            for col in 0..cols {
                let x = col * step;
                let y = row * step;
                
                let actual_width = chunk_size.min(self.width - x);
                let actual_height = chunk_size.min(self.height - y);
                
                let chunk_data = self.read_region(decoded_buffer, x, y, actual_width, actual_height)?;
                callback(x, y, actual_width, actual_height, &chunk_data)?;
                
                if (row * cols + col) % 10 == 0 {
                    println!("已处理: {}/{} 块", row * cols + col + 1, rows * cols);
                }
            }
        }
        
        println!("分块处理完成!");
        Ok(())
    }
    
    /// 切图到文件
    pub fn extract_tiles(&self, decoded_buffer:&[u8],tile_width: u32, tile_height: u32, output_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(output_dir)?;
        
        self.process_in_chunks(decoded_buffer,tile_width, 0, |x, y, width, height, data| {
            print!("width::{},height::{},data::{}",width,height,data.len());
            let img = image::RgbImage::from_raw(width, height, data.to_vec())
                .ok_or_else(|| anyhow::anyhow!("无法创建图片"))?;
            
            let filename = format!("tile_{}_{}.png", y / tile_height, x / tile_width);
            let filepath = output_dir.join(filename);
            
            img.save(&filepath)
                .with_context(|| format!("保存失败: {}", filepath.display()))?;
            
            Ok(())
        })
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
    
    // TIFF signatures: "II*\0" (little endian) or "MM\0*" (big endian)
    // if data.starts_with(&[0x49, 0x49, 0x2A, 0x00]) || 
    //    data.starts_with(&[0x4D, 0x4D, 0x00, 0x2A]) {
    //     return Ok(ImageFormat::Tiff);
    // }
    
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
                png::ColorType::Indexed => {
                    // 索引色PNG：本质是1通道（调色板索引），若需映射为RGB则返回3
                    // 这里优先返回实际存储的通道数（1），而非映射后的3
                    1
                }
            };
            // 额外校验：位深度是否合法（可选，增强健壮性）
            if !matches!(info.bit_depth, png::BitDepth::Eight | png::BitDepth::Sixteen) {
                print!(
                    "PNG位深度非8/16位：{}，可能影响后续处理",
                    info.bit_depth as u8
                );
            }
            Ok((info.width, info.height, channels))
        },
        ImageFormat::Jpeg => {
            let cursor = Cursor::new(data);
            let mut decoder = jpeg_decoder::Decoder::new(cursor);
            let _metadata = decoder.read_info()?;
            // let (width,height) = decoder.info()?;
            let (width,height,channels) = match decoder.info() {
                Some(info) => (info.width as u32,info.height as u32, info.pixel_format.pixel_bytes() as u8),
                None => (0,0,3),
            };
            println!("====================channels:{}",channels);
            Ok((width, height, channels))
        },
        // ImageFormat::Tiff => {
        //     let cursor = Cursor::new(data);
        //     let mut decoder = tiff::decoder::Decoder::new(cursor)?;
        //     let (width, height) = decoder.dimensions()?;
        //     let channels = 3; // 简化处理
        //     Ok((width, height, channels))
        // },
    }
}

 
#[derive(Debug, Clone, Copy, PartialEq)]
struct Rect { 
    pub x:f32,
    pub y:f32,
    pub width:f32,
    pub height:f32,
}
impl Rect {
    pub fn new(x:f32,y:f32,width:f32,height:f32)->Self {
        Self{x,y,width,height}
    }
}
#[derive(Debug, Clone, Deserialize)]
struct LabelmeShape { 
    pub points:Vec<Vec<f32>>,
    pub shape_type:String,
}
#[derive(Debug, Clone, Deserialize)]
struct IgnoreConfig { 
    shapes:Vec<LabelmeShape>,
}
fn load_labelme_rects(path: &Path)-> Result<Vec<Rect>,Box<dyn std::error::Error>> { 
       let json_content = std::fs::read_to_string(path)?;
    let ignor_config: IgnoreConfig = serde_json::from_str(&json_content)?;
    let rects = ignor_config.shapes.iter()
    .filter(|shape| shape.shape_type == "rectangle")
    .map(|shape| {
         // 检查 points 长度是否为 2（矩形需要两个对角点）
            if shape.points.len() != 2 {
                return Err(format!(
                    "矩形形状的 points 数量错误，需要 2 个点，实际 {} 个",
                    shape.points.len()
                )
                .into());
            }

            // 检查每个点是否包含 x/y 两个坐标
            let point1 = &shape.points[0];
            let point2 = &shape.points[1];
            if point1.len() != 2 || point2.len() != 2 {
                return Err("点坐标格式错误，需要 [x,y] 格式".into());
            }

            let x1 = point1[0];
            let y1 = point1[1];
            let x2 = point2[0];
            let y2 = point2[1];

            // 计算宽度和高度（处理 x2 < x1 或 y2 < y1 的情况）
            let width = (x2 - x1).abs();
            let height = (y2 - y1).abs();
            // 确保 Rect 的 x/y 是左上角坐标
            let x = x1.min(x2);
            let y = y1.min(y2);
            Ok(Rect::new(x, y, width, height))
    }).collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    Ok(rects)
}   

fn main() -> Result<()> {
    // 创建流式处理器
    let processor = StreamingImageProcessor::from_file("big.jpg")?;
    
    let (width, height, channels) = processor.info();
    println!("图片信息: {}x{}, {} 通道", width, height, channels);
    
    let decoded_buffer = processor.decode_jpg()?;

    // 加载ignore区域(logo)
    let ignore_rects = load_labelme_rects(Path::new("./configs/roi_ignore/618_ignore.json"));
    println!("igonre区域：：：{:?}",ignore_rects);

    //加载 识别区域
    let rects = load_labelme_rects(Path::new("./configs/roi_merged/618.json")).unwrap();
    println!("detect区域：：：{:?}",rects);

    //加载模型
       // 检查模型文件是否存在
    let model_path = "unet.onnx";
    let _session = Session::builder()?
    .edit_from_file(model_path)?;


    
    //  let env = Environment::builder()
    //     .with_name("rust_onnx_demo")
    //     .with_log_level(LoggingLevel::Warning) // 日志级别：Warning/Info/Debug
    //     .build()?;
    // //  创建会话（加载模型）
    // let _session = env.new_session_builder()?
    //     .with_optimization_level(GraphOptimizationLevel::Basic)? // 优化级别
    //     .with_number_threads(1)? // 推理线程数
    //     .with_model_from_file(Path::new("./unet.onnx"))?;

    rects.iter().for_each(|rect| {
        println!("detect区域：：：{:?}",rect);
        // 读取特定区域
        // let region_data = processor.read_region(&decoded_buffer,rect.x as i32, rect.y as i32, rect.width as i32, rect.height as i32)?;
        // 获取一个1024x1024的图片
        let region_data = processor.read_region(&decoded_buffer,rect.x as u32, rect.y as u32, 1024,1024).unwrap();
        
        // let input_data = Tensor::new(&region_data)?;

    });
    // 方式1: 读取特定区域
    // println!("开始读取 [(1000,1000),(1512,1512)] 矩形区域: ");
    // let region_data = processor.read_region(&decoded_buffer,1000, 1000, 512, 512)?;
    // println!("读取了 512x512 区域，数据大小: {} KB", region_data.len()/1024);
    
    
    
    
    // 方式2: 分块切图
    // processor.extract_tiles(&decoded_buffer,1024, 1024, Path::new("./output"))?;
    


    // // 方式3: 自定义分块处理
    // processor.process_in_chunks(512, 64, |x, y, w, h, data| {
    //     println!("处理块: ({}, {}) {}x{}, 数据: {} bytes", x, y, w, h, data.len());
        
    //     // 在这里可以进行自定义处理：
    //     // - 图像分析
    //     // - 格式转换
    //     // - 压缩
    //     // - 发送到其他系统等
        
    //     Ok(())
    // })?;
    
    Ok(())
}

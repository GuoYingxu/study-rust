use std::fs::File;
use std::io::{Cursor,Read,Seek};
use std::time::Instant;
fn load_image_to_memery(path: &str) -> Result<Vec<u8>, Box<std::io::Error>> { 
    let seg = Instant::now();
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    println!("加载图片耗时{:?}ms,内存占用大小{}MB", seg.elapsed().as_millis(), buffer.len()/1024/1024);
    Ok(buffer)
}
fn crop_img(img_bytes: &[u8])->Result<Vec<u8>,Box<dyn std::error::Error>> {
    let mut img = image::load(Cursor::new(img_bytes), image::ImageFormat::Jpeg)?;
    let img2 = img.crop(100, 100, 100, 100);
    let gray_img = img2.to_luma8();
    let mut buffer = Vec::new();
    let mut cursor = Cursor::new(&mut buffer);
    gray_img.write_to(&mut cursor, image::ImageFormat::Jpeg)?;
    Ok(buffer)
}
fn main()->Result<(), Box<dyn std::error::Error>> { 
  let buffer = load_image_to_memery("big.jpg")?;
  print!("buffer 长度{:?}", buffer.len());
  let crop_buffer = crop_img(buffer.as_slice())?;
  print!("crop_buffer 长度{:?}", crop_buffer.len());
  Ok(())
}
use image::{GrayImage};
use std::time::Instant;

fn load_image(path: &str) -> Result<GrayImage, Box<dyn std::error::Error>> {
    let img = image::open(path)?;
    let gray_img = img.to_luma8();
    println!("{:?}", gray_img.dimensions());
    Ok(gray_img)
}
fn main() {
    let seg =Instant::now();
    let _img = load_image("image.png").unwrap();
    let spendtime = seg.elapsed();
    println!("耗时：：{}ms", spendtime.as_millis());
}

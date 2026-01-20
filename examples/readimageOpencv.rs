// use opencv::{
//     core::Scalar,
//     highgui::{imshow, wait_key, WINDOW_NORMAL},
//     imgcodecs::imread,
//     imgproc::{cvt_color, COLOR_BGR2GRAY},
//     prelude::*,
// }; 

fn main() -> Result<(), opencv::Error> { 
    // 读取图片（替换为你的图片路径，建议用绝对路径，比如 "D:/test.jpg"）
    // let img = imread(".jpg", imgcodecs::IMREAD_COLOR)?;
    // if img.empty() {
    //     return Err(opencv::Error::new(
    //         opencv::core::StsError,
    //         "无法读取图片，请检查路径",
    //     ));
    // }
    // // 转换为灰度图
    // let mut gray_img = Mat::default();
    // cvt_color(&img, &mut gray_img, COLOR_BGR2GRAY, 0)?;

    // // 创建窗口并显示图片
    // imshow("原始图片", &img)?;
    // imshow("灰度图片", &gray_img)?;

    // // 等待按键（0 表示无限等待，按任意键关闭窗口）
    // wait_key(0)?;
  Ok(())
}
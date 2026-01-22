use ort::session::Session;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting ONNX Runtime example with ort...");

    // 检查模型文件是否存在
    let model_path = "unet.onnx";
    if !Path::new(model_path).exists() {
        return Err(format!("Model file does not exist: {}", model_path).into());
    }
    println!("Found model file: {}", model_path);

    let _session = Session::builder()?
        .edit_from_file(model_path)?;
    println!("Session created successfully");
    println!("Model loaded successfully!");

    Ok(())

}
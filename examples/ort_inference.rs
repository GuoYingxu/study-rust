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

    let session = Session::builder()?
        .edit_from_file(model_path)?;

    println!("Session created successfully");
    println!("Model loaded successfully!");

    // 打印输入和输出信息
    println!("Number of inputs: {}", session.inputs().len());
    println!("Number of outputs: {}", session.outputs().len());

    for (i, input) in session.inputs().iter().enumerate() {
        println!("Input {}: name={}", i, input.name());
    }

    for (i, output) in session.outputs().iter().enumerate() {
        println!("Output {}: name={}", i, output.name());
    }

    Ok(())
}
use onnxruntime::{environment::Environment, GraphOptimizationLevel, LoggingLevel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing ONNX Runtime initialization...");

    // 创建环境
    let env = Environment::builder()
        .with_name("simple_test")
        .with_log_level(LoggingLevel::Warning) // 使用较低的日志级别减少输出
        .build()?;

    println!("Environment created successfully");

    // 尝试从文件加载模型
    // 注意：我们暂时不指定模型文件，而是先测试环境是否正常工作
    match std::fs::metadata("unet.onnx") {
        Ok(metadata) => {
            println!("Model file exists, size: {} bytes", metadata.len());
            
            // 现在尝试创建会话
            let session = env.new_session_builder()?
                .with_optimization_level(GraphOptimizationLevel::Basic)?
                .with_number_threads(1)?
                .with_model_from_file("unet.onnx")?;
                
            println!("Session created successfully!");
        },
        Err(e) => {
            eprintln!("Model file 'unet.onnx' not accessible: {}", e);
            return Err(Box::new(e));
        }
    };

    Ok(())
}
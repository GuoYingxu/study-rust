use onnxruntime::{
    environment::Environment,
    GraphOptimizationLevel, LoggingLevel,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing ONNX Runtime environment initialization only...");

    let env = Environment::builder()
        .with_name("test_env_only")
        .with_log_level(LoggingLevel::Warning)
        .build()?;

    println!("Environment created successfully - no crash occurred!");
    println!("This means the onnxruntime library is properly linked.");

    Ok(())
}
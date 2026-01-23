use anyhow::Result;
use ndarray::Array4;
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use std::time::Instant;

fn print_gpu_info() -> i32 {
    println!("=== GPU 设备信息 ===");

    let mut gpu_count = 0;

    // 先尝试使用简单的 --list-gpus 命令
    match std::process::Command::new("nvidia-smi")
        .arg("--list-gpus")
        .output()
    {
        Ok(output) => {
            if output.status.success() {
                let gpu_info = String::from_utf8_lossy(&output.stdout);
                let lines: Vec<&str> = gpu_info.lines().filter(|l| !l.is_empty()).collect();
                gpu_count = lines.len() as i32;

                if gpu_count == 0 {
                    println!("警告: 未检测到 NVIDIA GPU");
                } else {
                    println!("检测到 {} 张 NVIDIA GPU:\n", gpu_count);
                    for line in lines {
                        println!("  {}", line);
                    }
                    println!();

                    // 获取每个GPU的详细信息
                    for i in 0..gpu_count {
                        if let Ok(detail) = std::process::Command::new("nvidia-smi")
                            .args(&[
                                "--query-gpu=name,memory.total,driver_version",
                                "--format=csv,noheader,nounits",
                                "-i",
                                &i.to_string(),
                            ])
                            .output()
                        {
                            if detail.status.success() {
                                let info = String::from_utf8_lossy(&detail.stdout);
                                let parts: Vec<&str> = info.trim().split(',').collect();
                                if parts.len() >= 3 {
                                    println!("  [GPU {} 详情]", i);
                                    println!("    名称: {}", parts[0].trim());
                                    println!("    显存: {} MB", parts[1].trim());
                                    println!("    驱动版本: {}", parts[2].trim());
                                    println!();
                                }
                            }
                        }
                    }
                }
            } else {
                println!("警告: 无法获取 GPU 信息");
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !stderr.is_empty() {
                    println!("错误信息: {}", stderr);
                }
            }
        }
        Err(e) => {
            println!("警告: nvidia-smi 命令未找到，请确认已安装 NVIDIA 驱动");
            println!("错误: {}", e);
        }
    }

    // 检查是否有集成显卡或其他显卡
    println!("注意: nvidia-smi 只显示 NVIDIA GPU。");
    println!("      如果有 Intel/AMD 集成显卡，它们不会显示在这里。\n");

    gpu_count
}

fn print_current_gpu_usage(device_id: i32) {
    // 获取特定GPU的当前使用情况
    match std::process::Command::new("nvidia-smi")
        .args(&[
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            &device_id.to_string(),
        ])
        .output()
    {
        Ok(output) => {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                let parts: Vec<&str> = info.trim().split(',').collect();
                if parts.len() >= 5 {
                    println!("     GPU 名称: {}", parts[0].trim());
                    println!("     显存使用: {} MB / {} MB", parts[1].trim(), parts[2].trim());
                    println!("     GPU 利用率: {}%", parts[3].trim());
                    println!("     温度: {}°C", parts[4].trim());
                }
            }
        }
        Err(_) => {
            println!("     无法获取 GPU 使用信息");
        }
    }
}

fn main() -> Result<()> {
    let total_start = Instant::now();  // 总计时开始

    println!("=== ONNX Runtime GPU 推理测试 ===");
    println!("提示: 使用 'cargo run --release --example simple_onnx_test' 获得最佳性能\n");

    // 打印 GPU 信息并获取可用 GPU 数量
    let gpu_count = print_gpu_info();

    // 检查模型文件
    println!("1. 检查模型文件...");
    let model_path = "unet.onnx";
    match std::fs::metadata(model_path) {
        Ok(metadata) => {
            println!("   ✓ 模型文件: {}", model_path);
            println!("   ✓ 文件大小: {:.2} MB\n", metadata.len() as f64 / 1024.0 / 1024.0);
        }
        Err(e) => {
            eprintln!("   ✗ 无法访问模型文件 '{}': {}", model_path, e);
            return Err(e.into());
        }
    }

    // 创建会话并启用 CUDA
    println!("2. 创建推理会话 (启用 CUDA GPU 加速)...");

    // 设置要使用的 GPU 设备 ID（0 = 第一张显卡，1 = 第二张显卡）
    // 如果你有多张 NVIDIA GPU，修改此值来选择不同的 GPU
    let gpu_device_id = 0;  // 可选值: 0, 1, 2, ... (取决于你的 GPU 数量)

    if gpu_count == 0 {
        println!("   警告: 未检测到 NVIDIA GPU，将使用 CPU 执行推理");
    } else if gpu_device_id >= gpu_count {
        println!("   警告: GPU 设备 {} 不存在（共 {} 张 GPU），将使用 GPU 0", gpu_device_id, gpu_count);
    } else {
        println!("   → 选择使用 GPU 设备 ID: {} (共 {} 张可用)", gpu_device_id, gpu_count);
    }

    println!("\n   如需使用其他 GPU，请修改代码中的 gpu_device_id 变量");
    println!("   例如: let gpu_device_id = 1;  // 使用第二张 GPU\n");

    // 尝试启用 CUDA，如果失败则回退到 CPU
    let session_start = Instant::now();
    let mut session = Session::builder()?
        .with_execution_providers([
            CUDAExecutionProvider::default()
                .with_device_id(gpu_device_id)
                .with_memory_limit(2 * 1024 * 1024 * 1024) // 2GB
                .build(),
        ])?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?  // 增加线程数以提高性能
        .with_inter_threads(4)?
        .commit_from_file(model_path)?;
    let session_time = session_start.elapsed();

    println!("   ✓ 会话创建成功");
    println!("   ⏱  模型加载耗时: {:.2} ms", session_time.as_secs_f64() * 1000.0);
    println!("   ✓ 实际使用的 GPU 设备: {}", gpu_device_id);
    println!();

    // 显示当前GPU的详细信息
    println!("   当前 GPU 详细状态:");
    print_current_gpu_usage(gpu_device_id);
    println!();

    // 打印输入输出信息
    println!("\n3. 模型信息:");
    for (i, input) in session.inputs().iter().enumerate() {
        println!("   输入 {}: 名称={}", i, input.name());
    }
    for (i, output) in session.outputs().iter().enumerate() {
        println!("   输出 {}: 名称={}", i, output.name());
    }

    // 创建 512x512 单通道输入图片（与模型的 patch_size 匹配，性能更优）
    // 如需测试更大尺寸，可修改为 1024x1024
    println!("\n4. 准备输入数据...");
    let data_prep_start = Instant::now();

    let batch_size = 1;
    let channels = 1;
    let height = 1024;  // 使用 512 以匹配 unet.yaml 的 patch_size
    let width = 1024;

    println!("   输入尺寸: {}x{}x{} (batch, channels, height, width)",
             batch_size, channels, height);

    // 创建输入张量: [batch, channels, height, width]
    // 使用渐变数据模拟图片 (值范围 0.0 - 1.0)
    let mut input_data = Vec::with_capacity((batch_size * channels * height * width) as usize);
    for i in 0..(batch_size * channels * height * width) {
        // 创建简单的渐变模式用于测试
        let value = (i % 256) as f32 / 255.0;
        input_data.push(value);
    }

    let input_array = Array4::from_shape_vec(
        (batch_size as usize, channels as usize, height as usize, width as usize),
        input_data,
    )?;

    let data_prep_time = data_prep_start.elapsed();

    println!("   ✓ 输入张量形状: {:?}", input_array.shape());
    println!(
        "   ✓ 数据范围: [{:.3}, {:.3}]",
        input_array.iter().cloned().fold(f32::INFINITY, f32::min),
        input_array.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!("   ⏱  数据准备耗时: {:.2} ms", data_prep_time.as_secs_f64() * 1000.0);

    // 将 ndarray 转换为 ORT Value
    let tensor_start = Instant::now();
    let input_tensor = Value::from_array(input_array)?;
    let tensor_time = tensor_start.elapsed();
    println!("   ⏱  张量转换耗时: {:.2} ms", tensor_time.as_secs_f64() * 1000.0);

    // 执行推理
    println!("\n5. 执行推理...");

    // 推理前的 GPU 状态
    println!("   推理前 GPU 状态:");
    print_current_gpu_usage(gpu_device_id);

    println!();
    let inference_start = Instant::now();

    let outputs = session.run(ort::inputs![input_tensor])?;

    let inference_time = inference_start.elapsed();
    println!("   ✓ 推理完成");
    println!("   ⏱  纯推理耗时: {:.2} ms", inference_time.as_secs_f64() * 1000.0);
    println!("   ⏱  吞吐量: {:.2} FPS", 1000.0 / inference_time.as_millis() as f64);
    println!("   ⏱  单张图片延迟: {:.2} ms", inference_time.as_secs_f64() * 1000.0);

    // 推理后的 GPU 状态
    println!("\n   推理后 GPU 状态:");
    print_current_gpu_usage(gpu_device_id);

    // 打印推理结果
    println!("\n6. 推理结果:");
    let result_process_start = Instant::now();

    for (i, (_name, value)) in outputs.iter().enumerate() {
        // 提取输出张量
        let (shape, data) = value.try_extract_tensor::<f32>()?;

        println!("\n   输出 {} 信息:", i);
        println!("   - 形状: {:?}", shape);
        println!("   - 总元素数: {}", data.len());

        // 计算统计信息
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = data.iter().sum();
        let mean = sum / data.len() as f32;

        println!("   - 数值范围: [{:.6}, {:.6}]", min, max);
        println!("   - 平均值: {:.6}", mean);

        // 打印前10个值作为样本
        println!("   - 前10个值:");
        for (j, &value) in data.iter().take(10).enumerate() {
            println!("     [{:2}] = {:.6}", j, value);
        }

        // 如果是分割输出 (通常是 [batch, classes, height, width])
        let shape_dims = shape.as_ref();
        if shape_dims.len() == 4 && shape_dims[1] == 12 {
            println!("\n   检测到语义分割输出 (12类):");
            println!("   - Batch大小: {}", shape_dims[0]);
            println!("   - 类别数: {}", shape_dims[1]);
            println!("   - 输出尺寸: {}x{}", shape_dims[2], shape_dims[3]);

            // 计算每个类的平均响应
            println!("\n   各类别平均响应值:");
            let num_classes = shape_dims[1] as usize;
            let pixels_per_class = (shape_dims[2] * shape_dims[3]) as usize;

            for class_idx in 0..num_classes {
                let class_start = class_idx * pixels_per_class;
                let class_end = class_start + pixels_per_class;
                let class_data = &data[class_start..class_end];
                let class_mean: f32 = class_data.iter().sum::<f32>() / class_data.len() as f32;
                println!("     类别 {:2}: {:.6}", class_idx, class_mean);
            }
        }
    }

    let result_process_time = result_process_start.elapsed();
    let total_time = total_start.elapsed();

    // 打印时间统计总结
    println!("\n=== 性能统计总结 ===");
    println!("⏱  模型加载时间: {:.2} ms", session_time.as_secs_f64() * 1000.0);
    println!("⏱  数据准备时间: {:.2} ms", data_prep_time.as_secs_f64() * 1000.0);
    println!("⏱  张量转换时间: {:.2} ms", tensor_time.as_secs_f64() * 1000.0);
    println!("⏱  纯推理时间:   {:.2} ms  ← 核心性能指标", inference_time.as_secs_f64() * 1000.0);
    println!("⏱  结果处理时间: {:.2} ms", result_process_time.as_secs_f64() * 1000.0);
    println!("   ────────────────────────");
    println!("⏱  总耗时:       {:.2} ms", total_time.as_secs_f64() * 1000.0);
    println!();
    println!("📊 关键性能指标:");
    println!("   • 推理吞吐量: {:.2} FPS", 1000.0 / inference_time.as_millis() as f64);
    println!("   • 端到端延迟: {:.2} ms", total_time.as_secs_f64() * 1000.0);
    println!("   • GPU 利用率: 推理前后可见差异");

    println!("\n=== 推理测试完成 ===");
    Ok(())
}
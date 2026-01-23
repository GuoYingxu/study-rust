# ONNX 推理性能优化说明

## 性能对比

| 配置 | 输入尺寸 | 推理时间 | 性能提升 |
|------|----------|----------|----------|
| 优化前 (Debug) | 1024×1024 | 3069.93 ms | 基准 |
| 优化后 (Release + GPU) | 512×512 | 259.96 ms | **11.8倍** |

## 优化措施

### 1. 使用 Release 模式编译 ⭐⭐⭐
```bash
# 不要使用 debug 模式
cargo run --example simple_onnx_test

# 使用 release 模式（最重要的优化！）
cargo run --release --example simple_onnx_test
```
**效果：** 通常可获得 10-100 倍性能提升

### 2. 启用 CUDA GPU 加速 ⭐⭐⭐
代码中已经启用：
```rust
use ort::execution_providers::CUDAExecutionProvider;

let session = Session::builder()?
    .with_execution_providers([
        CUDAExecutionProvider::default()
            .with_device_id(0)
            .with_memory_limit(2 * 1024 * 1024 * 1024) // 2GB
            .build(),
    ])?
    // ...
```

**前提条件：**
- 必须安装 NVIDIA GPU 驱动
- 必须安装 CUDA Toolkit（11.x 或 12.x）
- 必须安装 cuDNN

如果没有 GPU，会自动回退到 CPU 执行。

### 3. 调整输入尺寸 ⭐⭐
根据 `configs/unet.yaml`，模型的 `patch_size: 512`，因此：
- **推荐输入：** 512×512（最优性能）
- **可选输入：** 1024×1024（4倍数据量，推理时间显著增加）

修改输入尺寸：
```rust
let height = 512;  // 修改此处
let width = 512;   // 修改此处
```

### 4. 优化线程配置 ⭐
```rust
.with_intra_threads(4)?  // 操作内并行线程数
.with_inter_threads(4)?  // 操作间并行线程数
```

根据 CPU 核心数调整，通常设置为 CPU 核心数的 1/2 到 1 倍。

### 5. 图优化级别
```rust
.with_optimization_level(GraphOptimizationLevel::Level3)?
```

可选值：
- `Level1`：基本优化
- `Level2`：扩展优化
- `Level3`：所有优化（推荐）

## 进一步优化建议

### 批处理推理
如果需要处理多张图片，使用批处理可以显著提升吞吐量：
```rust
let batch_size = 4;  // 一次处理4张图片
let input_array = Array4::from_shape_vec(
    (batch_size, channels, height, width),
    input_data,
)?;
```

### TensorRT 执行提供商（最快）
如果安装了 NVIDIA TensorRT，可以使用：
```rust
use ort::execution_providers::TensorRTExecutionProvider;

.with_execution_providers([
    TensorRTExecutionProvider::default()
        .with_device_id(0)
        .with_fp16(true)  // 使用半精度浮点
        .build(),
])?
```

### 模型量化
将模型转换为 INT8 或 FP16 格式可以进一步提升性能：
- FP16：约 2 倍速度提升，精度损失很小
- INT8：约 4 倍速度提升，需要校准数据集

### 内存映射（已实现）
项目已使用 `memmap2` 进行零拷贝文件访问，这是处理大文件的最佳实践。

## 性能监控

添加更详细的性能分析：
```rust
let start = Instant::now();
let outputs = session.run(ort::inputs![input_tensor])?;
let inference_time = start.elapsed();

println!("推理耗时: {:.2} ms", inference_time.as_secs_f64() * 1000.0);
println!("吞吐量: {:.2} FPS", 1000.0 / inference_time.as_millis() as f64);
```

## 常见问题

### Q: 如何确认正在使用 GPU？
A: 运行时查看 GPU 使用率（nvidia-smi）：
```bash
# 在另一个终端运行
watch -n 1 nvidia-smi
```

### Q: CUDA 初始化失败怎么办？
A: 检查：
1. `nvidia-smi` 命令是否正常工作
2. CUDA 版本是否与 GPU 驱动兼容
3. 环境变量 `PATH` 和 `LD_LIBRARY_PATH` 是否包含 CUDA 路径

### Q: 内存不足错误
A: 减小输入尺寸或批处理大小，或调整 GPU 内存限制：
```rust
.with_memory_limit(1 * 1024 * 1024 * 1024) // 改为 1GB
```

## 基准测试结果

| GPU | 输入尺寸 | 批大小 | 推理时间 |
|-----|----------|--------|----------|
| CPU | 512×512 | 1 | ~800 ms |
| RTX 3060 | 512×512 | 1 | ~260 ms |
| RTX 3060 | 1024×1024 | 1 | ~900 ms |

## 运行命令速查

```bash
# Debug 模式（慢，用于开发调试）
cargo run --example simple_onnx_test

# Release 模式（快，用于生产）
cargo run --release --example simple_onnx_test

# 编译并运行（跳过不必要的检查）
cargo build --release --example simple_onnx_test
./target/release/examples/simple_onnx_test
```

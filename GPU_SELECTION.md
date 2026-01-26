# GPU 选择和使用指南

## 当前系统 GPU 配置

根据检测结果，你的系统有：
- **1 张 NVIDIA GPU:** GeForce GTX 1050 Ti (4096 MB 显存)
- 可能还有集成显卡（Intel/AMD），但 nvidia-smi 无法检测非 NVIDIA 显卡

## 如何选择使用哪张 GPU

### 方法 1: 修改代码中的 GPU 设备 ID

在 `examples/simple_onnx_test.rs` 文件中，找到这一行：

```rust
let gpu_device_id = 0;  // 0 = 第一张 NVIDIA GPU, 1 = 第二张, ...
```

修改这个值来选择不同的 GPU：
- `0` - 使用第一张 NVIDIA GPU（默认）
- `1` - 使用第二张 NVIDIA GPU（如果有）
- `2` - 使用第三张 NVIDIA GPU（如果有）

### 方法 2: 使用环境变量

如果不想修改代码，可以使用 CUDA 环境变量：

```bash
# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES="0"
cargo run --release --example simple_onnx_test

# Windows CMD
set CUDA_VISIBLE_DEVICES=0
cargo run --release --example simple_onnx_test

# Linux/Mac
CUDA_VISIBLE_DEVICES=0 cargo run --release --example simple_onnx_test
```

**设置多个GPU：**
```bash
# 只使用 GPU 0 和 GPU 2（跳过 GPU 1）
$env:CUDA_VISIBLE_DEVICES="0,2"

# 反转 GPU 顺序（GPU 1 变成 0，GPU 0 变成 1）
$env:CUDA_VISIBLE_DEVICES="1,0"
```

## 当前检测到的 GPU 信息

程序启动时会显示所有可用的 NVIDIA GPU：

```
=== GPU 设备信息 ===
检测到 1 张 NVIDIA GPU:

  GPU 0: NVIDIA GeForce GTX 1050 Ti (UUID: GPU-b8c5fbf3-8977-fc81-5c6c-db6912e50632)

  [GPU 0 详情]
    名称: NVIDIA GeForce GTX 1050 Ti
    显存: 4096 MB
    驱动版本: 561.17
```

## 推理时的 GPU 监控

程序会在推理前后显示 GPU 状态：

```
推理前 GPU 状态:
  GPU 名称: NVIDIA GeForce GTX 1050 Ti
  显存使用: 2048 MB / 4096 MB
  GPU 利用率: 2%
  温度: 42°C

推理后 GPU 状态:
  GPU 名称: NVIDIA GeForce GTX 1050 Ti
  显存使用: 2048 MB / 4096 MB
  GPU 利用率: 4%
  温度: 42°C
```

## 关于多 GPU 系统

如果你说有 2 张显卡，可能的情况：

### 情况 1: 一张 NVIDIA + 一张集成显卡
- NVIDIA GPU 会被 ONNX Runtime 使用
- 集成显卡（Intel/AMD）用于显示输出
- **当前情况很可能是这种**

### 情况 2: 两张 NVIDIA GPU
- 两张都会显示在 GPU 设备列表中
- 可以通过修改 `gpu_device_id` 来选择使用哪一张
- 示例：
  ```
  检测到 2 张 NVIDIA GPU:
    GPU 0: NVIDIA GeForce GTX 1050 Ti
    GPU 1: NVIDIA GeForce RTX 3060
  ```

### 情况 3: 一张 NVIDIA + 一张 AMD
- nvidia-smi 只会显示 NVIDIA GPU
- ONNX Runtime 的 CUDA 执行提供者只能使用 NVIDIA GPU
- AMD GPU 需要使用 ROCm 执行提供者（需要单独配置）

## 如何查看所有 GPU

### 使用 nvidia-smi 查看 NVIDIA GPU
```bash
nvidia-smi --list-gpus
```

### 使用设备管理器（Windows）
1. 按 `Win + X` 打开菜单
2. 选择"设备管理器"
3. 展开"显示适配器"
4. 可以看到所有 GPU（包括集成显卡）

### 使用任务管理器（Windows）
1. 按 `Ctrl + Shift + Esc` 打开任务管理器
2. 切换到"性能"选项卡
3. 可以看到所有 GPU 及其实时使用情况

## GPU 内存不足怎么办？

如果遇到 GPU 内存不足的错误，可以：

### 1. 减小输入尺寸
```rust
let height = 256;  // 从 512 改为 256
let width = 256;
```

### 2. 调整 GPU 内存限制
```rust
CUDAExecutionProvider::default()
    .with_device_id(gpu_device_id)
    .with_memory_limit(1 * 1024 * 1024 * 1024)  // 从 2GB 改为 1GB
    .build()
```

### 3. 使用 CPU 执行
如果 GPU 内存太小，可以移除 CUDA 执行提供者，使用 CPU：

```rust
// 注释掉这部分
// .with_execution_providers([
//     CUDAExecutionProvider::default()
//         .with_device_id(gpu_device_id)
//         .with_memory_limit(2 * 1024 * 1024 * 1024)
//         .build(),
// ])?
```

## 性能对比

在你的 GTX 1050 Ti 上：
- **512×512 输入，GPU:** ~260-340 ms
- **512×512 输入，CPU:** ~800-1000 ms
- **性能提升:** 约 3 倍

## 多 GPU 并行推理（高级）

如果有多张 GPU，可以实现并行推理：

```rust
use rayon::prelude::*;

// 为每张 GPU 创建单独的会话
let sessions: Vec<_> = (0..gpu_count)
    .map(|gpu_id| {
        Session::builder()
            .with_execution_providers([
                CUDAExecutionProvider::default()
                    .with_device_id(gpu_id)
                    .build(),
            ])
            .commit_from_file("unet.onnx")
    })
    .collect();

// 并行处理多个输入
let results: Vec<_> = inputs.par_iter()
    .enumerate()
    .map(|(i, input)| {
        let gpu_id = i % gpu_count;  // 循环分配到不同 GPU
        sessions[gpu_id].run(input)
    })
    .collect();
```

## 故障排查

### GPU 未被使用（仍在用 CPU）
1. 检查 CUDA 是否正确安装：`nvcc --version`
2. 检查 cuDNN 是否正确安装
3. 检查环境变量 `PATH` 是否包含 CUDA 路径
4. 查看程序输出是否有 CUDA 相关错误信息

### GPU 利用率低
1. 增大 batch size（一次处理多张图片）
2. 增大输入尺寸（512 → 1024）
3. 使用 FP16 精度（需要 TensorRT）

### 想要更高性能
1. 使用 TensorRT 执行提供者（需要安装 TensorRT）
2. 模型量化（INT8/FP16）
3. 使用更强大的 GPU（RTX 系列）

## 总结

**当前系统推荐配置：**
- GPU Device ID: `0` (GTX 1050 Ti)
- 输入尺寸: `512×512`
- Batch Size: `1`
- 预期性能: ~260-340 ms/推理

如果未来添加了第二张 NVIDIA GPU，只需修改 `gpu_device_id` 即可切换使用。

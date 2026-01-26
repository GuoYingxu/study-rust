# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust-based image processing and machine learning inference system focused on memory-efficient processing of extremely large images (10+ MB) using ONNX models. The project uses a memory-mapped streaming approach to read only necessary image regions rather than loading entire files into memory.

**Key capabilities:**
- Streaming image processing for PNG and JPEG formats
- ONNX model inference (specifically UNet semantic segmentation)
- Region-of-Interest (ROI) detection and annotation support (Labelme format)
- Interactive GUI image viewer using egui
- Tensor conversion pipeline for ML inference

## Build and Run Commands

### Building
```bash
# Build all examples
cargo build --examples

# Build a specific example
cargo build --example region_to_tensor

# Build with optimizations (release mode)
cargo build --release --examples
```

### Running Examples
The main functionality is in the `examples/` directory, not `src/main.rs`:

```bash
# Core examples:
cargo run --example region_to_tensor        # Image region to ONNX tensor conversion
cargo run --example readbigimage           # Large image streaming with ROI processing
cargo run --example ort_inference          # ONNX model loading and inference
cargo run --example readimage              # egui-based image viewer GUI

# Testing/debugging:
cargo run --example ort_test               # Test ORT API functionality
cargo run --example env_test               # Environment debugging
cargo run --example useonnx                # Inspect ONNX model structure
```

### Other Commands
```bash
# Check for errors without building
cargo check

# Run with release optimizations
cargo run --release --example <name>

# Clean build artifacts
cargo clean
```

**Note:** This project does not have traditional unit tests. Testing is done through the example binaries listed above.

## Architecture

### Code Organization

- **`src/main.rs`** - Placeholder entry point (not the main code location)
- **`examples/`** - All functional code is organized as runnable examples
  - `region_to_tensor.rs` - Core streaming image processor implementation
  - `readbigimage.rs` - Large image processing with ROI support
  - `ort_inference.rs` - ONNX Runtime integration
  - `readimage.rs` - GUI application for image viewing
- **`configs/`** - Configuration and annotation data
  - `unet.yaml` - Model parameters (patch_size: 512, stride: 256, num_classes: 12)
  - `roi_merged/` - ROI annotation files (Labelme JSON format)
  - `roi_ignore/` - Regions to ignore (logos, watermarks, etc.)
- **`unet.onnx`** - Pre-trained UNet model (7.8 MB)
- **`big.jpg`** / **`image.png`** - Test images for development

### Key Architectural Patterns

#### StreamingImageProcessor
The core abstraction for memory-efficient image processing:

```rust
pub struct StreamingImageProcessor {
    format: ImageFormat,
    width: u32,
    height: u32,
    channels: u8,
    file_data: Arc<Mmap>,  // Memory-mapped file, no pixel data loaded
}
```

**Design principles:**
- Uses `memmap2` for zero-copy file access
- Reads only metadata initially (width, height, channels)
- Supports selective region reading via `read_region(x, y, width, height)`
- Different strategies: PNG (row-by-row extraction) vs JPEG (full decode then extract)

#### Image Processing Pipeline
1. Memory-map the image file
2. Detect format from file signature (magic bytes)
3. Read metadata only (fast, ~50µs)
4. Extract specific regions on demand (1024×1024 or 512×512 patches)
5. Convert to normalized f32 tensors (÷ 255, CHW format)
6. Feed to ONNX Runtime for inference

#### ROI Management
- Uses Labelme JSON format for annotations
- Two types: detection regions (`roi_merged/`) and ignore regions (`roi_ignore/`)
- Enables selective processing of image areas

### Key Dependencies

| Library | Purpose | Notes |
|---------|---------|-------|
| `ort` (2.0.0-rc.11) | ONNX Runtime bindings | ML inference engine |
| `ndarray` | N-dimensional arrays | Tensor operations |
| `egui/eframe` (0.28) | GUI framework | Desktop app with image display |
| `memmap2` | Memory-mapped I/O | Zero-copy file access |
| `image` | General image ops | High-level API |
| `png` / `jpeg-decoder` | Format-specific decoders | Low-level format parsing |
| `rayon` | Data parallelism | Parallel processing (when needed) |
| `serde/serde_json` | Serialization | Config and ROI loading |

### Data Flow

```
File on disk
  ↓ (memmap2)
Memory-mapped view (no data loaded yet)
  ↓ (format detection + metadata parse)
Image dimensions & format
  ↓ (read_region)
Raw pixel data (u8) for specific ROI
  ↓ (normalize ÷ 255)
f32 tensor (CHW format)
  ↓ (ort::Session)
ONNX inference
  ↓
Segmentation output (12 classes)
```

## Development Guidelines

### When Adding Image Format Support
- Implement format detection in `detect_format()` using magic bytes
- Add metadata reading for the new format in `read_image_metadata()`
- Implement region extraction strategy in `read_region()` method
- Consider memory efficiency: can you extract regions without full decode?

### When Working with ONNX Models
- Model configuration lives in `configs/unet.yaml`
- Use `cargo run --example useonnx` to inspect model structure
- Tensor format must be CHW (channels, height, width), not HWC
- Input normalization: pixel values must be f32 in range [0.0, 1.0]

### Memory Considerations
- The streaming processor design avoids loading full images
- For JPEG, full decode is currently required (format limitation)
- PNG supports true streaming via row-by-row reading
- Always check actual memory usage via the processor's diagnostic prints

### Configuration Files
- YAML for model hyperparameters (`configs/unet.yaml`)
- JSON for ROI annotations (Labelme format in `configs/roi_merged/` and `configs/roi_ignore/`)
- ROI files use image filenames as keys (e.g., `618.json` for processing `618.jpg`)

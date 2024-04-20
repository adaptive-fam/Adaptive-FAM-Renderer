## Adaptive-FAM: *GPU-Accelerated Out-of-Core Volume Rendering with Functional Approximation and Adaptive Encoding for Interactive Exploration*

![results](https://github.com/adaptive-fam/Adaptive-FAM/blob/main/flame_blocks_small.png)

### Introduction
This repo contains the code of the renderer that generates visualization results through decoding micro-models encoded by Adaptive-FAM. The rendering framework supports:

- Multi-resolution for faster loading of content of interest
- Out-of-core caching for handling large-scale input data
- GPU-acceleration through CUDA kernel functions
- Prefetching techniques

###  Dependencies
- C++11 or higher compiler.
- [CUDA](https://developer.nvidia.com/cuda-toolkit), NVIDIA CUDA Toolkit.
- [GLUT](https://www.opengl.org/resources/libraries/glut/), OpenGL Utility Toolkit

### Build
```bash
git clone https://github.com/adaptive-fam/Adaptive-FAM-Renderer
cd Adaptive-FAM-Renderer
make
```

### Run
Usage: sudo volumeRender [sample distance] [prefetching options] [encoding methods]
- sample distance: The distance between contiguous samples on the ray of the ray casting volume rendering
- prefetching options: 
	- lru: LRU caching policy without prefetching (default)
	- appa: LRU caching policy with APPA prefetching method
	- markov: LRU caching policy with ForeCache prefetching method
	- lsrm: LRU caching policy with LSTM prefetching method
- encoding methods:
	- ds: Using discrete micro-blocks encoded with down sampling with ghost area (default)
	- mfa: Using continuous micro-blocks encoded with Adaptive-FAM
Run the renderer with root privilege for the cache droping action

### Output
- Screenshots of each visualization rendering for quality evaluation. Images are save in .tga format in folder named "screenShot".
- Benchmark profile, named benchmark.txt, for performance evaluation.

### Change Input
When rendering input micro-blocks or micro-models of a new dataset, modify the following paths in volumeRender.cpp:
- volumeFilePath: Path to the micro-blocks with ghost area encoded using down sampling
- mfaFilePath: Path to the micro-models encoded using Adaptive-FAM
- camera_trajectory: Path to the new user trajectory
- camera_trajectory_predict: Path to the predicted user trajectory using prefetching methods, ForeCache or LSTM

### Acceleration Techniques
- Find visible blocks: A lookup table, named cornerBlockMap.txt, is generated to avoid the exhausive search for the process of finding visible blocks. Please refer to the code for how to generate the lookup table.
- Update cache: Utilizing low level memory operations to minimize caching latency for a given cache miss rate.
- Mutithreading: Rendering and prefetching are handled on different threads at the same time for parallel process.

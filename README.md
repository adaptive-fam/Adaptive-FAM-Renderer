## Adaptive-FAM: *GPU-Accelerated Out-of-Core Volume Rendering with Functional Approximation and Adaptive Encoding for Interactive Exploration*

![results](https://github.com/adaptive-fam/Adaptive-FAM/blob/main/flame_blocks_small.png)

### Introduction
This repo contains the code of the renderer that generates visualization results through decoding micro-models encoded by Adaptive-FAM. The rendering framework supports:

-Multi-resolution for faster loading of content of interest.

-Out-of-core caching for handling large-scale input data.

-GPU-acceleration through CUDA kernel functions.

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
Usage: volumeRender [sample distance] [prefetching options] [encoding methods]
- sample distance: The distance between contiguous samples on the ray of the ray casting volume visualization technique
- prefetching options: 
	- lru: LRU caching policy without prefetching
	- appa: LRU caching policy with APPA prefetching method
	- markov: LRU caching policy with ForeCache prefetching method
	- lsrm: LRU caching policy with LSTM prefetching method
- encoding methods:
	- dis: Using discrete micro-blocks encoded with down sampling with ghost area
	- mfa: Using continuous micro-blocks encoded with Adaptive-FAM

### Output
- Screenshots of each visualization rendering for quality evaluation. Images are save in .tga format in folder named "screenShot".
- Benchmark profile, named benchmark.txt, for performance evaluation.

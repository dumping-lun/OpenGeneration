# OpenGeneration

欢迎来到我们的OpenGenaration项目,这也是msc科研部2025级的第一个项目！该项目旨在通过在 MNIST 数据集上完成一个**带条件**的图像生成任务，帮助大家学习和实践深度生成模型。

本项目将首先推荐 **条件变分自编码器 (CVAE)** 来实现可控的数字图像生成。完成此部分后，欢迎有兴趣的同学挑战更前沿的 **扩散模型 (Diffusion Model)**。

## 目录

- [项目简介](#项目简介)
- [任务要求](#任务要求)
- [快速开始](#快速开始)
- [技术背景](#技术背景)
  - [条件变分自编码器 (CVAE)](#条件变分自编码器-cvae)
  - [进阶挑战：扩散模型](#进阶挑战扩散模型)
- [文件结构](#文件结构)
- [参考资料](#参考资料)

## 项目简介

本项目的核心任务是，训练一个模型，当我们向模型输入一个指定的数字标签（如：`7`）时，模型能够生成一张看起来像手写数字 `7` 的图片。这个任务将帮助我们理解如何将条件信息融入到生成过程中，从而实现可控的、有目标的图像生成。

### 项目基本架构
```
OpenGeneration/
├── models/
│   ├── __init__.py
│   └── cvae.py
├── scripts/
│   └── train_cvae.py
├── generate_cvae.py
└── README.md
```
## 任务要求

1.  **理解代码**: 你可以根据项目的codebase来修改 `models/cvae.py` 和 `scripts/train_cvae.py` 中的算法代码以实现训练需求，或者自己搭建框架。
2.  **模型训练**: 运行训练脚本 `scripts/train_cvae.py`，训练你自己的 CVAE 模型。
3.  **条件生成**: 运行 `generate_cvae.py` 脚本，加载你训练好的模型，生成指定数字的图像，并观察生成效果。
4.  **以pr的方式提交你的项目到我们的主仓库**:


## 快速开始

1.  **创建项目**: (在此之前希望大家熟悉[git](https://liaoxuefeng.com/books/git/introduction/index.html))
    ```bash
    git clone https://github.com/MicrosoftClub-Xidian-Research/OpenGeneration.git
    cd OpenGeneration
    ```

2.  **创建环境**: 希望大家已经熟悉用[conda](https://zhuanlan.zhihu.com/p/94744929)创建环境了。

3.  **安装依赖**: 建议创建一个虚拟环境。
    ```bash
    conda create -n gen python=3.9
    conda activate gen
    pip install torch torchvision numpy matplotlib tqdm
    ```

4.  **训练模型**:
    ```bash
    # 这将开始训练，并会在项目根目录下生成 cvae_mnist.pth 文件
    python scripts/train_cvae.py
    ```

5.  **生成图像**:
    ```bash
    # 待训练完成后，运行此脚本查看生成效果
    python generate_cvae.py
    ```

## 技术背景

### 条件变分自编码器 (CVAE)

**简介**:
条件变分自编码器（CVAE）是变分自编码器（VAE）的扩展。VAE 通过编码器将图片压缩到低维的“潜在空间”，再通过解码器从这个空间中重构图片。CVAE 的巧妙之处在于，它在编码和解码时都加入了**条件信息**（在本项目中，就是数字的标签 0-9），从而让我们可以控制解码器生成特定内容的图像。

**推荐阅读**:
*   **论文**: *VAE* - [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)  
    *CVAE* - [This File](https://proceedings.neurips.cc/paper_files/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf)
*   **博客**: *Conditional Variational Autoencoders* - [https://ijdykeman.github.io/ml/2016/12/21/cvae.html](https://ijdykeman.github.io/ml/2016/12/21/cvae.html)

### 进阶挑战：扩散模型

**简介**:
扩散模型是近年来在图像生成领域取得巨大成功的模型（如 Stable Diffusion）。它通过模拟一个“加噪”和“去噪”的过程来生成图像。首先，它不断地对清晰图片加入噪声，直到图片变成完全的随机噪声；然后，训练一个神经网络来学习逆转这个过程，即从噪声中逐步恢复出清晰的、真实的图片。通过在去噪的每一步中加入条件信息，就可以控制最终生成的图像内容。

**推荐阅读**:
*   **论文**: *Denoising Diffusion Probabilistic Models (DDPM)* - [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
*   **代码**: *Conditional_Diffusion_MNIS* - [https://github.com/TeaPearce/Conditional_Diffusion_MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST)


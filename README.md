# Fine-Tuning LLaMA 3 8B with Unsloth on AWS `g4dn.2xlarge` EC2 instance

This repository demonstrates fine-tuning Meta's LLaMA 3 8B model using the Unsloth framework on an AWS g4dn.2xlarge instance.

## What is Unsloth?

Unsloth is an open-source library designed to accelerate the fine-tuning and training of large language models (LLMs). By manually optimizing compute-intensive operations and crafting custom GPU kernels, Unsloth achieves significant improvements in training speed and memory efficiency without requiring hardware modifications. Unsloth is 10x faster on a single GPU and up to 32x faster on multiple GPU systems compared to Flash Attention 2 (FA2).
We support NVIDIA GPUs from Tesla T4 to H100, and we’re portable to AMD and Intel GPUs.

## Environment Setup

To run the provided Jupyter notebook, set up a Python 3.11.11 environment with PyTorch installed. Ensure that your environment meets these requirements before executing the notebook.

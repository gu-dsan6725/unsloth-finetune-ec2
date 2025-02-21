# Fine-Tuning LLaMA 3 8B with `Unsloth` on AWS `g6e.12xlarge` EC2 instance

This repository demonstrates fine-tuning Meta's LLaMA 3 8B model using the Unsloth framework on an AWS `g6e.12xlarge` EC2 instance. View the instructions below to launch the EC2 instance, clone this repository and run the example notebook.

## What is `Unsloth`?

`Unsloth` is an open-source library designed to accelerate the fine-tuning and training of large language models (LLMs). By manually optimizing compute-intensive operations and crafting custom GPU kernels, `Unsloth` achieves significant improvements in training speed and memory efficiency without requiring hardware modifications. According to the `Unsloth` documentation provided [here](https://docs.unsloth.ai/#:~:text=Unsloth%20makes%20finetuning%20large%20language%20models%20like%20Llama%2D3%2C%20Mistral%2C%20Phi%2D4%20and%20Gemma%202x%20faster%2C%20use%2070%25%20less%20memory%2C%20and%20with%20no%20degradation%20in%20accuracy!), it makes finetuning large language models like Llama-3, Mistral, Phi-4 and Gemma 2x faster, use 70% less memory, and with no degradation in accuracy. `Unsloth` support NVIDIA GPUs from Tesla T4 to H100, and we’re portable to AMD and Intel GPUs.

## Prerequisites

To run the provided Jupyter notebook, follow the steps below to create an EC2 instance where you can clone this repository, set up the conda kernel, install the required packages and run the example notebook.

1. Create an EC2 instance with the `Deep Learning OSS Nvidia Driver AMI GPU PyTorch` AMI and `g6e.12xlarge` instance type. You can see step by step instructions [here](https://aws-samples.github.io/foundation-model-benchmarking-tool/misc/ec2_instance_creation_steps.html).
    - At the time of this writing the AMI used for us-east-1 was ami-067bd563cecc90173.
    - 450GB of gp3 storage with 16000 IOPS and throughput set to 1000.
    ![ec2-storage](images/ec2-storage-config.png)

1. SSH to your EC2 instance and clone this repo.

    ```bash
    git clone https://github.com/madhurprash/unsloth-finetune-ec2.git
    cd unsloth-finetune-ec2
    ```

1. If you are looking to train a gated model, such as the [Llama3.1 8b instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model, then provide an `HF` token that has access to the gated repository. If you are training a model that is not gated, then the `HF` token is not required.

    ```bash
    export HF_TOKEN=your_hf_token
    ```

1. Install `uv` and Python dependencies:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ```

1. Restore Python virtual environment from the pyproject.toml file.

    ```bash
    uv venv && source .venv/bin/activate && uv pip sync pyproject.toml
    ```

1. Create a conda kernel. This is needed to run the notebook on your EC2 instance:

    ```bash
    uv add zmq
    python -m ipykernel install --user --name=.venv --display-name="Python (uv env)"
    ```

1. Open the JupyterLab notebook to run and select the `Python (uv env)` if not selected automatically (if it is not, refresh the page or close the notebook and open it again and you should see it selected or be able to select it from the drop down).

## Steps to run the solution


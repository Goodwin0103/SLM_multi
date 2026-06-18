#!/bin/bash
# ODNN 项目服务器环境一键搭建脚本
# 在服务器上每个用户在自己的账号下执行一次
# Usage: bash setup_server.sh
set -e

echo "=== ODNN Server Setup ==="

# 1. 初始化 conda（适配自建 miniconda3，不用 module load）
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "ERROR: conda not found at ~/miniconda3 or ~/anaconda3"
    exit 1
fi

# 2. 创建 conda 环境
echo "Creating conda environment 'odnn'..."
conda create -n odnn python=3.11 -y

# 3. 安装依赖
echo "Installing Python dependencies..."
conda activate odnn
pip install torch numpy scipy matplotlib pandas mat73

# 4. 克隆项目（请替换为你的实际仓库地址）
echo "Cloning project..."
if [ -d ~/odnn_project ]; then
    echo "  ~/odnn_project already exists, skipping git clone."
else
    cd ~
    git clone https://github.com/YOUR_REPO/ODNN.git odnn_project
fi

# 5. 创建工作目录结构
echo "Creating workspace directories..."
mkdir -p ~/odnn_workspace/{uploads,runs}

echo ""
echo "=== Setup complete ==="
echo "Activate environment: source ~/miniconda3/etc/profile.d/conda.sh && conda activate odnn"
echo "Project directory:   ~/odnn_project"
echo "Workspace directory: ~/odnn_workspace"

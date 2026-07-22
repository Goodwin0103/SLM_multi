# 一、本地电脑配置

## 1.1 克隆仓库

```bash
git clone -b Jinsong https://github.com/Goodwin0103/SLM_multi.git ODNN

```

如果电脑上没装git，打开 [https://github.com/Goodwin0103/SLM_multi/tree/Jinsong](https://github.com/Goodwin0103/SLM_multi/tree/Jinsong) 下载ZIP后解压

## 1.2 安装 Python 依赖

在项目根目录下：

```bash
cd frontend
pip install -r requirements.txt

```

如果本地电脑也想跑训练（Local 模式），还需要install mat73、h5py。

## 1.3 启动前端

在项目根目录下:

```bash
streamlit run frontend/app.py

```

浏览器打开:
http://localhost:8501。

# 二、远程服务器配置

## 2.1 安装Miniconda

在服务器根目录(比如 /home/jslai):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

~/miniconda3/bin/conda init bash

source ~/.bashrc

```

## 2.2 创建 conda 环境并安装依赖

```bash
conda create -n odnn python=3.11 -y

```

(
如果遇到需要accept的报错提示:)

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

激活虚拟环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh

conda activate odnn

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install numpy scipy matplotlib pandas mat73

```

## 2.3 克隆项目代码到服务器

在服务器根目录(比如 /home/jslai):

```bash
git clone -b Jinsong https://github.com/Goodwin0103/SLM_multi.git ODNN

```

## 2.4 创建工作目录

在服务器根目录(比如 /home/jslai):

```bash
mkdir -p ~/odnn_workspace/{uploads,runs}

```

注: `~/odnn_workspace` 是服务器上所有训练输出（config、log、checkpoint、metrics）的根目录。每次 Remote 训练会自动在下面建 `runs/run_YYYYMMDD_HHMMSS/` 子目录，互不覆盖。

## 2.5 MATLAB 数据集生成

如果需要在服务器上用 MATLAB 生成新的 .mat 模式数据集，需要把整个整个eigenmodes_generation_grin目录放到服务器根目录下。
在本地电脑终端:

```bash
scp -r ~/路径/eigenmodes_generation_grin jslai@141.30.127.3:~/

```

输出.mat数据集目录：`~/eigenmodes_generation_grin/mmf_data`

# 三、本地电脑配置免密SSH连接

## 3.1 本地电脑生成密钥:

```bash
ssh-keygen -t ed25519

```

## 3.2 把公钥复制到服务器

```bash
ssh-copy-id <你的用户名>@<你的服务器IP>

```

比如: `ssh-copy-id jslai@141.30.127.3`

## 3.3 验证

```bash
ssh <用户名>@<IP> echo OK

```

成功输出 OK，且不需要输密码

## 3.4 本地电脑开启SSH连接复用

```bash
mkdir -p ~/.ssh/sockets
nano ~/.ssh/config

```

填入:

```text
Host 141.30.127.3
        ControlMaster auto
        ControlPath ~/.ssh/sockets/%r@%h-%p
        ControlPersist 600

```

按Ctrl O保存 然后Ctrol X退出

# 四、前端 Settings 页面配置

打开 http://localhost:8501，左侧导航进 Settings，修改Username为服务器上用户名
检查其它是否正确，点 Save，再点 Test Connection，成功会显示 "Connection successful"

这些配置保存在 `~/.odnn/remote_config.json`（不在项目目录里，不会被 git 提交）

# 五、批量训练多个modes

在服务器ODNN目录下，有一个`batch_train_wl.py`文件可以实现批量训练多个modes，最后把指标画成折现图。

使用方法:

1. 在服务器ODNN目录下修改batch_train_wl.py里对应的参数
2. 把训练用的.mat数据集放入到ODNN目录，同时修改MAT_FILE路径
3. 在ODNN目录下打开终端，进入虚拟环境 `conda avtivate odnn`
执行: `nohup python batch_train_wl.py > batch.log 2>&1 &`
4. 训练的结果会显示在ODNN/results/batch_sweep目录下
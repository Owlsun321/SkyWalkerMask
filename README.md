# SkyWalkerMask
基于GroundingDINO和Segment Anything Model (SAM)的天空移除和人物掩码生成工具。

## 功能特性
- ☁️ **天空移除**: 智能检测并移除图像中的天空区域
- 🎯 **自动人物检测**: 使用GroundingDINO检测图像中的人物
- 🎭 **精确掩码生成**: 基于SAM生成高质量的人物分割掩码
- 📁 **批量处理**: 支持整个目录的批量图像处理
- 🖼️ **多格式支持**: 支持JPG、PNG、BMP、TIFF等常见图像格式
- ⚡ **GPU加速**: 支持CUDA加速，提升处理速度

## 环境要求
- Python 3.8+
- CUDA 11.2+ (推荐，用于GPU加速)
- 8GB+ 内存
- 2GB+ 显存 (使用GPU时)

## 安装步骤
### 1. 创建虚拟环境
conda create -n skywalker python=3.9 -y
conda activate skywalker

### 2. 安装PyTorch
# CUDA版本 (推荐)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# CPU版本
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu

### 3. 下载模型文件
由于模型文件较大，需要单独下载：

```bash
# 创建模型目录
mkdir -p models

# 下载 GroundingDINO 模型 (约1.2GB)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O models/groundingdino_swint_ogc.pth

# 下载 SAM 模型 (约2.4GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam_vit_h_4b8939.pth

# 下载天空移除模型 (约50MB)
# 注意：这个模型需要从原始项目获取，或使用其他天空移除模型
```

**或者手动下载：**
- [GroundingDINO 模型](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
- [SAM 模型](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- 天空移除模型需要从相关项目获取

### 4. 安装GroundingDINO
cd SkyWalkerMask
cd GroundingDINO
pip install -e .
cd ..

### 5. 安装其他依赖
pip install -r requirements.txt

## 使用方法
#### Windows (命令提示符)
```cmd
curl -L -o models/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
curl -L -o models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
curl -L -o models/model.zip https://github.com/OpenDroneMap/SkyRemoval/releases/download/v1.0.6/model.zip
powershell -command "Expand-Archive -Path 'models/model.zip' -DestinationPath 'models' -Force"
del models\model.zip
```

#### Windows (PowerShell)
```powershell
Invoke-WebRequest -Uri "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" -OutFile "models/groundingdino_swint_ogc.pth"
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -OutFile "models/sam_vit_h_4b8939.pth"
Invoke-WebRequest -Uri "https://github.com/OpenDroneMap/SkyRemoval/releases/download/v1.0.6/model.zip" -OutFile "models/model.zip"
Expand-Archive -Path "models/model.zip" -DestinationPath "models" -Force
Remove-Item "models/model.zip"
```

#### Linux/macOS
```bash
wget -O models/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -O models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -O models/model.zip https://github.com/OpenDroneMap/SkyRemoval/releases/download/v1.0.6/model.zip
unzip models/model.zip
rm models/model.zip
```

#### 手动下载
如果命令行下载失败，可以手动下载以下文件到 `models/` 目录：

- **GroundingDINO模型**: https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
- **SAM模型**: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
- **天空移除模型**: https://github.com/OpenDroneMap/SkyRemoval/releases/download/v1.0.6/model.zip (需要解压)


## 使用方法
### 1. 准备数据

将要处理的图像文件放入 `data` 目录：

SkyWalkerMask/
├── data/           # 输入图像目录
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── outputs/        # 输出掩码目录 (自动创建)
├── human_mask.py   # 人物掩码生成脚本
└── skyremoval.py   # 天空移除脚本

### 2. 运行脚本
conda activate skywalker
python human_mask.py
python skyremoval.py

### 3. 查看结果
处理完成后，黑白掩码文件将保存在 `outputs` 目录中：


## 项目结构
SkyWalkerMask/
├── README.md              # 说明文档
├── requirements.txt       # 依赖列表
├── human_mask.py         # 人物掩码生成脚本
├── skyremoval.py         # 天空移除脚本
├── data/                 # 输入图像目录
├── outputs/              # 输出掩码目录
├── models/               # 模型权重目录
│   ├── groundingdino_swint_ogc.pth  # GroundingDINO权重
│   ├── sam_vit_h_4b8939.pth         # SAM权重
│   └── model.onnx                   # 天空移除模型
└── GroundingDINO/        # GroundingDINO源码目录
    ├── groundingdino/    # 核心代码
    ├── setup.py          # 安装脚本
    └── requirements.txt  # GroundingDINO依赖

## 参数说明
脚本会自动处理 `data` 目录中的所有图像文件，生成对应的黑白掩码。

### 支持的图像格式
- JPG/JPEG, PNG, BMP, TIFF

### 输出格式
- 掩码文件保存为JPG格式，黑白二值图像

### GPU加速
确保安装了CUDA版本的PyTorch，脚本会自动检测并使用GPU。

### 内存优化
处理大图像时可能需要更多内存，可以调整批处理大小或图像分辨率。

## 技术栈
- **GroundingDINO**: 零样本目标检测
- **Segment Anything Model (SAM)**: 图像分割
- **SkyRemoval**: 天空区域检测和移除
- **Supervision**: 检测和分割工具库
- **OpenCV**: 图像处理
- **PyTorch**: 深度学习框架

## 许可证
本项目基于MIT许可证开源。

## 💘 Acknowledgements
本项目基于以下优秀的开源项目构建，感谢这些项目的贡献者：
- **[Segment Anything](https://github.com/facebookresearch/segment-anything)** - Meta的通用图像分割模型
- **[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)** - IDEA-Research的开放词汇目标检测模型
- **[Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)** - 结合GroundingDINO和SAM的完整解决方案
- **[SkyRemoval](https://github.com/OpenDroneMap/SkyRemoval)** - OpenDroneMap的天空移除工具

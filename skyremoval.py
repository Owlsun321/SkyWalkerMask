import argparse
import time
import numpy as np
import cv2
import os
import onnx
import onnxruntime as ort
import glob
import unicodedata
import re

# 全局变量配置
VERSION = "v1.0.6"
DEFAULT_MODEL_FOLDER = "models"
GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPS = 20, 0.01

# 使用GPU如果可用，否则使用CPU
PROVIDER = "CUDAExecutionProvider" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPUExecutionProvider"

def slugify(value, allow_unicode=False):

    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def find_model_file():
    """查找模型文件"""
    # 获取第一个.onnx扩展名的文件，比较简单的方式
    candidates = glob.glob(os.path.join(DEFAULT_MODEL_FOLDER, '*.onnx'))
    if len(candidates) == 0:
        raise Exception('No model found (expected at least one file with .onnx extension')
    return candidates[0]

def box(img, radius):
    """基于快速引导滤波的box函数"""
    dst = np.zeros_like(img)
    (r, c) = img.shape

    s = [radius, 1]
    c_sum = np.cumsum(img, 0)
    dst[0:radius+1, :, ...] = c_sum[radius:2*radius+1, :, ...]
    dst[radius+1:r-radius, :, ...] = c_sum[2*radius+1:r, :, ...] - c_sum[0:r-2*radius-1, :, ...]
    dst[r-radius:r, :, ...] = np.tile(c_sum[r-1:r, :, ...], s) - c_sum[r-2*radius-1:r-radius-1, :, ...]

    s = [1, radius]
    c_sum = np.cumsum(dst, 1)
    dst[:, 0:radius+1, ...] = c_sum[:, radius:2*radius+1, ...]
    dst[:, radius+1:c-radius, ...] = c_sum[:, 2*radius+1 : c, ...] - c_sum[:, 0 : c-2*radius-1, ...]
    dst[:, c-radius: c, ...] = np.tile(c_sum[:, c-1:c, ...], s) - c_sum[:, c-2*radius-1 : c-radius-1, ...]

    return dst

def guided_filter(img, guide, radius, eps):
    """引导滤波函数"""
    (r, c) = img.shape

    CNT = box(np.ones([r, c]), radius)

    mean_img = box(img, radius) / CNT
    mean_guide = box(guide, radius) / CNT

    a = ((box(img * guide, radius) / CNT) - mean_img * mean_guide) / (((box(img * img, radius) / CNT) - mean_img * mean_img) + eps)
    b = mean_guide - a * mean_img

    return (box(a, radius) / CNT) * img + (box(b, radius) / CNT)

class SkyFilter():
    """天空移除滤波器类"""

    def __init__(self, model=None, width=384, height=384):
        self.model = model or find_model_file()
        self.width, self.height = width, height

        print(' ?> 使用提供者 %s' % PROVIDER)
        self.load_model()

    def load_model(self):
        """加载模型"""
        if not os.path.exists(self.model):
            raise Exception(f'模型文件不存在: {self.model}')
            
        print(' -> 加载模型')
        onnx_model = onnx.load(self.model)

        # 检查模型
        try:
            onnx.checker.check_model(onnx_model)
        except onnx.checker.ValidationError as e:
            print(' !> 模型无效: %s' % e)
            raise
        else:
            print(' ?> 模型有效!')

        self.session = ort.InferenceSession(self.model, providers=[PROVIDER])


    def get_mask(self, img):
        """获取天空掩码"""
        height, width, c = img.shape

        # 调整图像大小以适应模型输入
        new_img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        new_img = np.array(new_img, dtype=np.float32)

        # ONNX模型的输入向量
        input = np.expand_dims(new_img.transpose((2, 0, 1)), axis=0)
        ort_inputs = {self.session.get_inputs()[0].name: input}

        # 运行模型
        ort_outs = self.session.run(None, ort_inputs)

        # 获取输出
        output = np.array(ort_outs)
        output = output[0][0].transpose((1, 2, 0))
        output = cv2.resize(output, (width, height), interpolation=cv2.INTER_LANCZOS4)
        output = np.array([output, output, output]).transpose((1, 2, 0))
        output = np.clip(output, a_max=1.0, a_min=0.0)

        return self.refine(output, img)

    def refine(self, pred, img):
        """精化掩码"""
        refined = guided_filter(img[:,:,2], pred[:,:,0], GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPS)

        res = np.clip(refined, a_min=0, a_max=1)
        
        # 将res转换为CV_8UC1
        res = np.array(res * 255., dtype=np.uint8)
        
        # 阈值化
        res = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY_INV)[1]
        
        return res

    def run_folder(self, folder, dest):
        """处理文件夹"""
        print(' -> 处理文件夹 ' + folder)

        # 如果存在尾随斜杠则移除
        if folder[-1] == '/':
            folder = folder[:-1]

        if os.path.exists(dest) is False:
            os.mkdir(dest)

        img_names = os.listdir(folder)

        start = time.time()

        # 过滤文件，只包含图像
        img_names = [name for name in img_names if os.path.splitext(name)[1].lower() in ['.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff']]

        for idx in range(len(img_names)):
            img_name = img_names[idx]
            print(' -> [%d / %d] 处理 %s' % (idx+1, len(img_names), img_name))
            self.run_img(os.path.join(folder, img_name), dest)

        expired = time.time() - start
                
        print('\n ?> 完成，耗时 %.2f 秒' % expired)
        if len(img_names) > 0:
            print(' ?> 每张图片平均耗时: %.2f 秒' % (expired / len(img_names)))
            print('\n ?> 输出保存在 ' + dest)
        else:
            print(' ?> 未找到图像')

    def run_img(self, img_path, dest):
        """处理单张图像"""
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img / 255., dtype=np.float32)

        mask = self.get_mask(img)
        
        img_name = os.path.basename(img_path)
        fpath = os.path.join(dest, img_name)

        fname, _ = os.path.splitext(fpath)
        mask_name = fname + '.jpg'
        cv2.imwrite(mask_name, mask)
        
        return mask_name

    def run(self, source, dest):
        """运行天空移除"""
        if os.path.exists(dest) is False:
            os.mkdir(dest)

        # 检查source是否是数组
        if isinstance(source, np.ndarray):
            for idx in range(len(source)):
                itm = source[idx]
                self.run(itm, dest)
        else:
            # 检查source是目录还是文件
            if os.path.isdir(source):
                self.run_folder(source, dest)
            else:
                print(' -> 处理: %s' % source)
                start = time.time()
                self.run_img(source, dest)
                print(" -> 完成，耗时 %.2f 秒" % (time.time() - start))
                print(' ?> 输出保存在 ' + dest)

# 命令行参数解析
parser = argparse.ArgumentParser(description='SkyRemoval - 天空移除工具')
parser.add_argument('input_dir', type=str, nargs='?', default='data', help='输入图像目录路径 (默认: data)')
parser.add_argument('output_dir', type=str, nargs='?', default='outputs', help='输出掩码目录路径 (默认: outputs)')
parser.add_argument('--model', type=str, help='模型文件路径 (默认: models/model.onnx)')
parser.add_argument('--width', type=int, default=384, help='训练模型输入宽度')
parser.add_argument('--height', type=int, default=384, help='训练模型输入高度')

if __name__ == '__main__':
    print('\n *** SkyRemoval - %s ***\n' % VERSION)

    args = parser.parse_args()

    filter = SkyFilter(args.model, args.width, args.height)
    filter.run(args.input_dir, args.output_dir)

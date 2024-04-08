# 参数：input_mask_dir 掩码图片目录，黑白图片
# 参数：input_orig_dir 原始图片目录，文件名要和掩码图片文件名一致，后缀可以不一样
# 输出：根据掩码图片裁剪原始图片，得到原图的裁剪图片

import argparse
import glob
import os
import os.path as osp
import cv2

from pathlib import Path
from paddleseg.utils import logger


def parse_args():
    parser = argparse.ArgumentParser(description='image crop.')
    parser.add_argument('--input_mask_dir', help='input annotated directory',type=str)
    parser.add_argument('--input_orig_dir', help='input annotated directory',type=str)
    return parser.parse_args()


def main(args):
    output_dir = osp.join(args.input_orig_dir, 'crop_dir')
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        print('Creating crop directory:', output_dir) # 在原图目录创建裁剪后输出目录
    for mask_img in glob.glob(osp.join(args.input_mask_dir, '*.png')):  # 遍历input_mask_dir目录所有png的图片
        print('read mask img from:', mask_img)
        mask_img_name = Path(mask_img).stem  # 获取文件名
        print('read orig img name:', args.input_orig_dir+mask_img_name+".jpg")
        # 读取原始图像和关键部位掩码图像
        original_image = cv2.imread(args.input_orig_dir+"/"+mask_img_name+".jpg")
        mask_image = cv2.imread(mask_img, cv2.IMREAD_GRAYSCALE)  # 假设是单通道的灰度图像
        # 查找掩码图像中的边界
        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 找到最大的边界
        max_contour = max(contours, key=cv2.contourArea)
        # 获取最大边界的边界框
        x, y, w, h = cv2.boundingRect(max_contour)
        # 从原始图像中裁剪关键部位
        key_part = original_image[y:y + h, x:x + w]
        # 保存裁剪出的关键部位
        cv2.imwrite(output_dir+"/"+mask_img_name+".jpg", key_part)
    logger.info(f'crop img is saved in {output_dir}')


if __name__ == '__main__':
    args = parse_args()
    main(args)

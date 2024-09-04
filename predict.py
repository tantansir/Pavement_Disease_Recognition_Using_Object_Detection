from ultralytics import YOLO
import os
import shutil
import torch


def predict(source_path, target_path):

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ori_txt_path = 'runs/detect/predict/labels'

    if not os.path.exists(ori_txt_path):
        os.makedirs(ori_txt_path)
    # 创建预测文件夹
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    # 加载模型文件
    model = YOLO('Final_Model.pt',task='detect')
    
    image_files = os.listdir(source_path)
    # 遍历待预测的每张图片
    for image_file in image_files:
        image_file_path = os.path.join(source_path,image_file)
        model.predict(image_file_path,save_txt=True,save=False,conf=0.05,iou=0.5,device='cpu')

        txt_file_name = os.path.splitext(image_file)[0] + ".txt"
        ori_txt_file_path = os.path.join(ori_txt_path, txt_file_name)
        if not os.path.exists(ori_txt_file_path):
            target_txt_file_path = os.path.join(target_path, txt_file_name)
            open(target_txt_file_path, "w").close()

    txt_list = os.listdir(ori_txt_path)
    for txt_file in txt_list:
        ori_txt_file = os.path.join(ori_txt_path, txt_file)
        target_txt_file = os.path.join(target_path, txt_file)
        shutil.move(ori_txt_file, target_txt_file)


if __name__ == '__main__':
    
    # 请在此处输入预测图片文件夹路径source_path和生成预测文件文件夹路径target_path
    source_path = 'image'
    target_path = 'result'
    
    predict(source_path, target_path)

    
        
    
        











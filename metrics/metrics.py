# from utils.class_registry import ClassRegistry
# from pytorch_fid.fid_score import calculate_fid_given_paths
# import torch


# metrics_registry = ClassRegistry()


# @metrics_registry.add_to_registry(name="fid")
# class FID:
#     def __init__(self):
#         pass

#     def __call__(self, orig_pth, synt_pth):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         fid = calculate_fid_given_paths([orig_pth, synt_pth],batch_size=16,  device=device, dims=2048)
#         return fid
    
#     def __str__(self):
#         return "FID"
    
#     def __repr__(self):
#         return "FID"
from utils.class_registry import ClassRegistry
from pytorch_fid.fid_score import compute_statistics_of_path, calculate_frechet_distance
import torch
from PIL import Image
import os
import shutil
import numpy as np
from pytorch_fid.inception import InceptionV3

metrics_registry = ClassRegistry()

def collect_images_from_class_folders(root_dir):
    image_paths = []
    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_path, img_file))
    return image_paths

def create_temp_folder_with_images(image_paths, temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    for img_path in image_paths:
        try:
            img_name = os.path.basename(img_path)
            temp_path = os.path.join(temp_dir, img_name)
            if not os.path.exists(temp_path):
                img = Image.open(img_path)
                img.save(temp_path)
        except Exception as e:
            print(f"Ошибка при обработке {img_path}: {e}")
    return temp_dir

@metrics_registry.add_to_registry(name="fid")
class FID:
    def __init__(self, test_pth):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_test_dir = "./temp_test_images"
        self.test_image_paths = collect_images_from_class_folders(test_pth)
        self.temp_test_dir = self._prepare_temp_folder(self.test_image_paths, self.temp_test_dir)
        self.dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model = InceptionV3([block_idx]).to(self.device)
        self.batch_size = 128
        self.test_statistics = compute_statistics_of_path(path=self.temp_test_dir, model=self.model, batch_size=self.batch_size, dims=self.dims, device = self.device)

    
    def _prepare_temp_folder(self, image_paths, temp_dir):
        return create_temp_folder_with_images(image_paths, temp_dir)
    
    
    def __call__(self, synt_pth):
        if os.path.isdir(synt_pth) and not any(os.path.isdir(os.path.join(synt_pth, d)) for d in os.listdir(synt_pth)):
            synt_mean, synt_cov = compute_statistics_of_path(
                synt_pth,
                model=self.model,
                batch_size=self.batch_size,
                dims=self.dims,
                device=self.device
            )
        else:
            temp_synt_dir = "./temp_synt_images"
            synt_image_paths = collect_images_from_class_folders(synt_pth)
            self._prepare_temp_folder(synt_image_paths, temp_synt_dir)
            
            synt_mean, synt_cov = compute_statistics_of_path(
                temp_synt_dir,
                model=self.model,
                batch_size=self.batch_size,
                dims=self.dims,
                device=self.device
            )
            
            shutil.rmtree(temp_synt_dir)

        fid = calculate_frechet_distance(
            self.test_statistics[0], self.test_statistics[1],
            synt_mean, synt_cov
        )
        return fid

    
    def __del__(self):
        shutil.rmtree(self.temp_test_dir, ignore_errors=True)
    
    def __str__(self):
        return "FID"
    
    def __repr__(self):
        return "FID"

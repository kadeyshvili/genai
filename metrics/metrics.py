import torch
from torchvision.models import inception_v3
from torchvision import transforms
from torch.nn import functional as F
from scipy import linalg
import numpy as np
from PIL import Image
import os
from utils.class_registry import ClassRegistry


metrics_registry = ClassRegistry()


@metrics_registry.add_to_registry(name="fid")
class FID:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, orig_path, synt_path):
        orig_features = self._compute_inception_activations(orig_path)
        synt_features = self._compute_inception_activations(synt_path)

        mu1, sigma1 = np.mean(orig_features, axis=0), np.cov(orig_features, rowvar=False)
        mu2, sigma2 = np.mean(synt_features, axis=0), np.cov(synt_features, rowvar=False)

        fid_score = self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_score

    def _compute_inception_activations(self, images_path):
        activations = []
        for img_file in os.listdir(images_path):
            img_path = os.path.join(images_path, img_file)
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(image).squeeze(0)

            # Resize logits to 8x8
            pred = F.adaptive_avg_pool2d(pred.unsqueeze(0), (1, 1)).squeeze()
            activations.append(pred.cpu().numpy())

        activations = np.array(activations)
        return activations

    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance."""
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():
            print("WARNING: fid computation produces singular product; adding eps to diagonal of cov estimates")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                max_imag_component = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {max_imag_component}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# Example usage within a trainer:
# fid_metric = FID(device='cuda')
# fid_score = fid_metric(orig_path='path/to/orig', synt_path='path/to/synt')
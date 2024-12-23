from PIL import Image
from torch.utils.data import Dataset
from utils.data_utils import make_dataset
from utils.class_registry import ClassRegistry
from torchvision import transforms


datasets_registry = ClassRegistry()


@datasets_registry.add_to_registry(name="base_dataset")
class BaseDataset(Dataset):
    def __init__(self, root):
        self.paths = make_dataset(root)
        self.transforms = transforms.Compose(
            [
                transforms.Resize((64, 64)),  # Resize
                transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
                transforms.ToTensor(),  # Convert to tensor (0, 1)
                transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
            ]
        )

    def __getitem__(self, ind):
        path = self.paths[ind]
        image = Image.open(path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return {"images": image}

    def __len__(self):
        return len(self.paths)


import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as TF


class PetDataset(torch.utils.data.Dataset):
    def __init__(self, root="data"):
        self.dataset = OxfordIIITPet(
            root=root,
            split="trainval",
            target_types="segmentation",
            download=True
        )

        self.transform_img = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        # ------------------------------
        # IMAGE
        # ------------------------------
        img = self.transform_img(img)

        # ------------------------------
        # MASK (CORREÇÃO IMPORTANTE)
        # ------------------------------
        # resize SEM interpolação (crítico!)
        mask = TF.resize(mask, (128, 128), interpolation=TF.InterpolationMode.NEAREST)

        # para numpy
        mask = np.array(mask)

        # Oxford Pet labels:
        # 1 = animal
        # 2 = border
        # 3 = background
        mask = (mask == 1).astype(np.float32)

        # para tensor (1 canal)
        mask = torch.tensor(mask).unsqueeze(0)

        return img, mask


# ==============================
# TESTE VISUAL (debug)
# ==============================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = PetDataset()
    img, mask = dataset[0]

    plt.subplot(1, 2, 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.title("Image")

    plt.subplot(1, 2, 2)
    plt.imshow(mask[0], cmap="gray")
    plt.title("Mask")

    plt.show()
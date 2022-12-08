import os

import torch.utils.data
from PIL import Image
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset


class Datasets(Dataset):
    def __init__(self, data_dir, image_size=256, train=True):
        self.data_dir = data_dir
        self.image_size = image_size

        if train:
            self.transform_resize = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])
            self.transform = transforms.Compose([
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform_resize = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")

        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        if image.width < self.image_size or image.height < self.image_size:
            return self.transform_resize(image)
        else:
            return self.transform(image)

    def __len__(self):
        return len(self.image_path)


class Datasets_AdaptCrop(Dataset):
    """
    Adapt Crop img size with n * 64, only for test RD performance!
    """
    def __init__(self, data_dir, image_size=256, train=True):
        self.data_dir = data_dir
        self.image_size = image_size

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")

        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        w_crop, h_crop = (image.width // 64) * 64, (image.height // 64) * 64
        transform_this = transforms.Compose([
                transforms.CenterCrop((w_crop, h_crop)),
                transforms.ToTensor(),
            ])
        return transform_this(image)

    def __len__(self):
        return len(self.image_path)


class Datasets_AdaptPad(Dataset):
    """
    Adapt Crop img size with n * 64, only for test RD performance!
    """
    def __init__(self, data_dir, image_size=256, train=True):
        self.data_dir = data_dir
        self.image_size = image_size

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")

        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        w_pad = (image.width // 64 + 1) * 64  - image.width if not image.width % 64 == 0 else 0
        h_pad = (image.height // 64 + 1) * 64 - image.height if not image.height % 64 == 0 else 0
        pad_L = w_pad // 2
        pad_R = w_pad - pad_L
        pad_T = h_pad // 2
        pad_B = h_pad - pad_T
        
        image = transforms.functional.pad(image, [pad_L, pad_T, pad_R, pad_B], padding_mode="constant", fill=1)
        image = self.to_tensor(image)
        return image

    def __len__(self):
        return len(self.image_path)


class Datasets_withName(Dataset):
    """
    Adapt Crop img size with n * 64, only for test RD performance!
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")

        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        image = self.to_tensor(image)
        return image, image_ori

    def __len__(self):
        return len(self.image_path)


if __name__ == "__main__":
    print("hello")
    # train_coco = Datasets("/home/DataSets/MSCOCO/train2017")
    # train_loader_coco = torch.utils.data.DataLoader(train_coco, num_workers=24,
    #         batch_size=128, shuffle=True)

    # print(len(train_loader_coco))

    # for imgs in train_loader_coco:
    #     imgs = imgs.cuda()

    test_clic = Datasets_AdaptPad("/home/DataSets/clic2020_professional")
    loader = torch.utils.data.DataLoader(test_clic, num_workers=0,
            batch_size=1, shuffle=False)
    print(len(loader))

    for imgs in loader:
        imgs = imgs.cuda()

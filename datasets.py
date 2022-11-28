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



if __name__ == "__main__":
    print("hello")
    train_coco = Datasets("/workspace/compression/DataSets/MSCOCO/train2017")
    train_loader_coco = torch.utils.data.DataLoader(train_coco, batch_size=8, shuffle=True)

    print(len(train_loader_coco))

    # for imgs in train_loader_coco:
    #     imgs = imgs.cuda()
    #     print(imgs.shape)
    #     break
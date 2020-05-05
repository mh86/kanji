from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import random
import pickle
from pathlib import Path
from torchvision import transforms


class KanjiDataset(Dataset):
    def __init__(self, main_dir,
                 cat2ind_file,
                 img_paths_pickle=None,
                 mode='train',
                 transform=transforms.ToTensor()):
                 # transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])):
        super(KanjiDataset, self).__init__()
        self.main_dir = main_dir
        if img_paths_pickle is not None:
            self.img_paths_pickle = img_paths_pickle
        self.cat2ind_file = cat2ind_file
        self.cat2ind = self._cat2ind_create()
        self.mode = mode
        self.transforms = transform
        self.image_paths, self.labels, self.counts = self.get_images_and_labels()

    def _cat2ind_create(self):
        df = pd.read_csv(self.cat2ind_file)
        return dict(zip(df['Categories'].values, df['Label'].values))

    def get_images_and_labels(self):
        if self.mode == 'split':
            image_paths = list(self.main_dir.rglob('*.png'))
            labels = np.array([self.cat2ind[item.parent.stem] for item in image_paths])
        else:
            with open(self.img_paths_pickle, 'rb') as fp:
                image_paths = pickle.load(fp)
                labels = np.array([self.cat2ind[item.parent.stem] for item in image_paths])
        counts = dict(zip(*np.unique(labels, return_counts=True)))

        return image_paths, labels, counts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        if self.mode == 'train':
            count_for_label = self.counts[label]
            # Getting boolean of where this label occurs for choosing anchor and positive image
            mask = self.labels == label
            if count_for_label == 1:
                anchor_image = self.get_image(index)
                positive_image = self.get_image(index)
            else:
                anchor_index, positive_index = random.sample(list(np.where(mask)[0]), 2)
                anchor_image = self.get_image(anchor_index)
                positive_image = self.get_image(positive_index)
            negative_index1, negative_index2 = random.sample(list(np.where(~mask)[0]), 2)
            # negative_index1 = random.sample(list(np.where(~mask)[0]), 1)
            negative_label1 = self.labels[negative_index1]
            negative_label2 = self.labels[negative_index2]
            negative_image1 = self.get_image(negative_index1)
            negative_image2 = self.get_image(negative_index2)

            return [anchor_image, positive_image, negative_image1, negative_image2], \
                   np.array([label, label, negative_label1, negative_label2])
            # return [anchor_image, positive_image, negative_image1], \
            #        np.array([label, label, negative_label1])
        elif self.mode == 'valid':
            anchor_image = self.get_image(index)
            return anchor_image, label

    def get_image(self, index):
        image = cv2.imread(str(self.image_paths[index]), cv2.IMREAD_GRAYSCALE)
        # image = Image.open(str(self.image_paths[index]))
        if self.transforms:
            image = self.transforms(image)
        return image

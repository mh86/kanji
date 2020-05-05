from torchvision import transforms
from pathlib import Path
# import cv2
from PIL import Image
# from sklearn.model_selection import StratifiedKFold


# main_path = Path(__file__).parents[2]
# data_path = 'data/kkanji2/'
# paths = (main_path / data_path).rglob('U+*')
# num_images = 140424


class Augment:
    def __init__(self, main_dir, aug_transforms=transforms.RandomAffine(20, shear=10), aug_imgs_to_add=4, threshold=5):
        # Say a class has one sample, 4 additional images are added to ensure a ratio of 80:20 train:test split.
        super(Augment, self).__init__()
        self.main_dir = main_dir
        self.threshold = threshold
        self.aug_transforms = aug_transforms
        self.aug_imgs_to_add = aug_imgs_to_add
        self.augment_low_sampled_classes()

    def augment_low_sampled_classes(self):
        for path in self.main_dir.rglob('U+*'):
            class_paths = list(path.rglob('*.png'))
            if len(class_paths) <= self.threshold:
                for img_path in class_paths:
                    self.augment(img_path)

    def augment(self, img_path):
        # img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = Image.open(str(img_path))
        for i in range(self.aug_imgs_to_add):
            img = self.aug_transforms(img)
            new_img_path = img_path.parent / (img_path.stem + '_' + str(i+1) + 'aug' + img_path.suffix)
            img.save(new_img_path)

    def visualize(self, img_path):
        img = Image.open(str(img_path))
        img._show()

    def del_aug_imgs(self):
        for aug in self.main_dir.rglob('*aug.png'):
            aug.unlink()


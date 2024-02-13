import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import functional as Fv


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        validation_split,
        num_workers,
        collate_fn=default_collate,
    ):
        self.shuffle = shuffle
        self.validation_split = validation_split

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.train_sampler, self.valid_sampler = self._split_sampler(
            self.validation_split
        )

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }
        super().__init__(sampler=self.train_sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        # no split performed
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0, "Validation set size should be at least 1."
            assert (
                split < self.n_samples
            ), "Validation set size should be at most equal to the number of samples."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loader for EfficientCapsNet
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        self.mnist_img_size = 28
        image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Lambda(self.random_rotate),
                # transforms.Lambda(self.random_shift),
                # transforms.Lambda(self.random_squish),
                # transforms.Lambda(self.random_erase),
            ]
        )
        label_transform = transforms.Lambda(self.one_hot_encode)
        self.dataset = datasets.MNIST(
            data_dir,
            train=training,
            download=True,
            transform=image_transform,
            target_transform=label_transform,
        )
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )

    def one_hot_encode(self, label, num_classes=10):
        one_hot = torch.zeros(num_classes)
        one_hot[label] = 1
        return one_hot

    def random_rotate(self, img):
        rand_vals = torch.clamp(
            torch.normal(0, 0.33, size=(2,)), min=-0.9999, max=0.9999
        )
        if rand_vals[1] > 0:
            return img
        else:
            angle = rand_vals[0] * 30  # degrees
            rot_mat = cv2.getRotationMatrix2D(
                center=(self.mnist_img_size / 2, self.mnist_img_size / 2),
                angle=int(angle),
                scale=1.0,
            )
            new_img = cv2.warpAffine(
                src=img.numpy(),
                M=rot_mat,
                dsize=(self.mnist_img_size, self.mnist_img_size),
            )
            new_img = torch.from_numpy(new_img).float()
            return new_img

    def random_shift(self, img):
        img = img.view(self.mnist_img_size, self.mnist_img_size)  # reshape

        # find non-zero columns and rows
        nonzero_x_cols = torch.nonzero(torch.sum(img, dim=0) > 0, as_tuple=True)[0]
        nonzero_y_rows = torch.nonzero(torch.sum(img, dim=1) > 0, as_tuple=True)[0]

        # calculate margins
        left_margin = torch.min(nonzero_x_cols)
        right_margin = self.mnist_img_size - torch.max(nonzero_x_cols) - 1
        top_margin = torch.min(nonzero_y_rows)
        bot_margin = self.mnist_img_size - torch.max(nonzero_y_rows) - 1

        # generate random directions and values
        rand_dirs = torch.rand(2)
        dir_idxs = torch.floor(rand_dirs * 2).int()
        rand_vals = torch.clamp(torch.abs(torch.normal(0, 0.33, size=(2,))), max=0.9999)

        # calculate shift amounts
        x_amts = [
            torch.floor(-1.0 * rand_vals[0] * left_margin.float()),
            torch.floor(rand_vals[0] * (1 + right_margin).float()),
        ]
        y_amts = [
            torch.floor(-1.0 * rand_vals[1] * top_margin.float()),
            torch.floor(rand_vals[1] * (1 + bot_margin).float()),
        ]
        x_amt = int(x_amts[dir_idxs[1]])
        y_amt = int(y_amts[dir_idxs[0]])

        # perform shift on image
        img = img.view(self.mnist_img_size * self.mnist_img_size)
        img = torch.roll(img, shifts=y_amt * self.mnist_img_size, dims=0)
        img = img.view(self.mnist_img_size, self.mnist_img_size)
        img = img.t()

        img = img.reshape(self.mnist_img_size * self.mnist_img_size)
        img = torch.roll(img, shifts=x_amt * self.mnist_img_size, dims=0)
        img = img.view(self.mnist_img_size, self.mnist_img_size)
        img = img.t()

        return img.view(1, self.mnist_img_size, self.mnist_img_size)

    def random_squish(self, img):
        rand_vals = torch.clamp(torch.abs(torch.normal(0, 0.33, size=(2,))), max=0.9999)
        width_mod = int((rand_vals[0] * (self.mnist_img_size / 4)).floor() + 1)
        offset_mod = int((rand_vals[1] * 2.0).floor())
        offset = (width_mod // 2) + offset_mod
        img = Fv.resize(img, [self.mnist_img_size, self.mnist_img_size - width_mod])
        img = Fv.pad(img, (offset, 0, offset_mod, 0))
        img = Fv.crop(img, 0, 0, self.mnist_img_size, self.mnist_img_size)
        return img

    def random_erase(self, img):
        """Randomly erase a 4x4 patch from the image."""
        rand_vals = torch.rand(2)
        x = int((rand_vals[0] * 19).floor() + 4)
        y = int((rand_vals[1] * 19).floor() + 4)
        patch = torch.zeros(4, 4)
        mask = F.pad(
            patch,
            (x, self.mnist_img_size - x - 4, y, self.mnist_img_size - y - 4),
            mode="constant",
            value=1,
        )
        img = img * mask
        return img

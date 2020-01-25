import os
import numpy as np
import torch
import torch.utils.data as data
import cv2 as cv


class PenisData(data.Dataset):
    """
    Class defined to handle the synthetic dataset
    derived from pytorch's Dataset class.
    """

    def __init__(self, root_path, mode, label):
        self.mode = mode
        self.root_dir_name = os.path.dirname(root_path)
        fine_folder = os.path.join(self.root_dir_name, "fine/")
        real_folder = os.path.join(self.root_dir_name, "real/")
        coarse_folder = os.path.join(self.root_dir_name, "coarse/")
        self.label = label

        if self.mode == "train":
            self.data_path = os.path.join(fine_folder, self.label)
            root, dir, files = next(os.walk(self.data_path))

            self.data_names = []
            for i in files:
                if not i.endswith(".txt"):
                    self.data_names.append(os.path.join(self.data_path, i))

            real_path = os.path.join(real_folder, self.label)
            real_root, _, real_names = next(os.walk(real_path))
            real_index = open(os.path.join(real_folder, "training_split.txt"))
            self.real_index = real_index.read().split(", ")
            
            poses_file = open(os.path.join(self.data_path, "poses.txt"))
            poses_file_real = open(os.path.join(real_path, "poses.txt"))
            self.poses = poses_file.readlines()
            poses_real = poses_file_real.readlines()
            
            for i in self.real_index:
                loc = real_names.index("real%d.png" % int(i))
                self.data_names.append(os.path.join(real_path, real_names[loc]))
                self.poses.append("%s" % poses_real[2*int(i)])
                self.poses.append("%s" % poses_real[2*int(i) + 1])

        if self.mode == "database":
            self.data_path = os.path.join(coarse_folder, self.label)
            root, _, files = next(os.walk(self.data_path))
            self.data_names = []

            for i in files:
                if not i.endswith(".txt"):
                    self.data_names.append(os.path.join(self.data_path, i))

            poses_file = open(os.path.join(self.data_path, "poses.txt"))
            self.poses = poses_file.readlines()

        if self.mode == "test":
            self.data_path = os.path.join(real_folder, self.label)
            root, _, files = next(os.walk(self.data_path))

            indices = list(range(0, 1178))
            real_index = open(os.path.join(real_folder, "training_split.txt"))
            self.real_index = real_index.read().split(", ")
            
            poses_file = open(os.path.join(self.data_path, "poses.txt"))
            all_poses = poses_file.readlines()
            self.poses = []

            for i in self.real_index:
                indices.remove(int(i))
            
            self.data_names = []
            
            for i in indices:
                loc = files.index("real%d.png" % int(i))
                self.data_names.append(os.path.join(self.data_path, files[loc]))
                self.poses.append("%s" % all_poses[2*int(i)])
                self.poses.append("%s" % all_poses[2*int(i) + 1])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            return [self[i] for i in range(index)]
        elif isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        elif isinstance(index, int):
            image = cv.imread(self.data_names[index])
            filename = self.data_names[index].split("/")[-1]
            loc = self.poses.index('# %s\n' % filename)
            pose_list = self.poses[loc+1].split()
            pose_quat = []

            for i in pose_list:
                pose_quat.append(float(i))
            # get the data from direct index
            return image, pose_quat

        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.data_names)


import os
import sys
import glob
import trimesh
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class DataSetsLoader:
    def __init__(self, num_points=2048, num_class=10, batch_size=32):
        self.num_points = num_points;
        self.num_class = num_class;
        self.batch_size = batch_size;
        self.data_dir = "./datasets/ModelNet10.zip"
        self.load_data();
        
        
    def load_data(self):
        #self.data_dir = tf.keras.utils.get_file(
        #    "modelnet.zip",
        #    "http://modelnet.cs.princeton.edu/ModelNet10.zip",
        #    extract=True,
        #)
        folder = os.getcwd()
        path = os.path.join(folder, "datasets/ModelNet10.zip")
        self.data_dir = tf.keras.utils.get_file(fname="modelnet.zip", origin=path, extract=True)
        self.data_dir = os.path.join(os.path.dirname(self.data_dir), "ModelNet10")
        
        
    def show_sample_data(self):
        mesh = trimesh.load(os.path.join(self.data_dir, "chair/train/chair_0001.off"))
        mesh.show()
        
        points = mesh.sample(self.num_points)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        ax.set_axis_off()
        plt.show()
        
            
    def augment(points, label):
        # jitter points
        points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
        # shuffle points
        points = tf.random.shuffle(points)
        return points, label


    def parse_dataset(self, num_points=2048):
        train_points = []
        train_labels = []
        test_points = []
        test_labels = []
        class_map = {}
        folders = glob.glob(os.path.join(self.data_dir, "[!README]*"))
        # its win32, maybe there is win64 too?
        is_windows = sys.platform.startswith('win')

        for i, folder in enumerate(folders):
            print("processing class: {}".format(os.path.basename(folder)))
            # store folder name with ID so we can retrieve later
            if is_windows:
                class_map[i] = folder.split("\\")[-1]
            else:
                class_map[i] = folder.split("/")[-1]
            print("class map {}".format(class_map[i]))
            # gather all files
            train_files = glob.glob(os.path.join(folder, "train/*"))
            test_files = glob.glob(os.path.join(folder, "test/*"))

            for f in train_files:
                train_points.append(trimesh.load(f).sample(num_points))
                train_labels.append(i)

            for f in test_files:
                test_points.append(trimesh.load(f).sample(num_points))
                test_labels.append(i)
        return (
            np.array(train_points),
            np.array(test_points),
            np.array(train_labels),
            np.array(test_labels),
            class_map,
        )
                                                          
import h5py
import pptk
import numpy as np
import time


def load_h5(h5_filename):
    """
    loading h5 file
    :param h5_filename: h5 file path
    :return: point cloud data (2048 * 3) and label corrosponding to point cloud
    """
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


labels = {
    0: "airplane",
    1: "bathtub",
    2: "bed",
    3: "bench",
    4: "bookshelf",
    5: "bottle",
    6: "bowl",
    7: "car",
    8: "chair",
    9: "cone",
    10: "cup",
    11: "curtain",
    12: "desk",
    13: "door",
    14: "dresser",
    15: "flower_pot",
    16: "glass_box",
    17: "guitar",
    18: "keyboard",
    19: "lamp",
    20: "laptop",
    21: "mantel",
    22: "monitor",
    23: "night_stand",
    24: "person",
    25: "piano",
    26: "plant",
    27: "radio",
    28: "range_hood",
    29: "sink",
    30: "sofa",
    31: "stairs",
    32: "stool",
    33: "table",
    34: "tent",
    35: "toilet",
    36: "tv_stand",
    37: "vase",
    38: "wardrobe",
    39: "xbox"
        }


file_path_visualization = "../data/modelnet40_ply_hdf5_2048/ply_data_test0.h5"

points, point_labels = load_h5(file_path_visualization)

points_reshape = points.reshape(-1, 2048, 3)
point_labels_reshape = point_labels.reshape(-1, 1)


poses = list()
poses.append([0, 0, 0, np.pi / 4, 0 * np.pi / 2, 5])
poses.append([0, 0, 0, np.pi / 4, 1 * np.pi / 2, 5])
poses.append([0, 0, 0, np.pi / 4, 2 * np.pi / 2, 5])
poses.append([0, 0, 0, np.pi / 4, 3 * np.pi / 2, 5])
poses.append([0, 0, 0, np.pi / 4, 4 * np.pi / 2, 5])

temp = True
vis = None
for i in range(points_reshape.shape[0]):
    if temp:
        vis = pptk.viewer(points_reshape[i])
        vis.set(point_size=0.01, r=5, theta=1.574)
        temp = False
    else:
        vis.load(points_reshape[i])
        vis.set(point_size=0.01, r=5, theta=1.574)
    print("point cloud label :: ", labels[point_labels_reshape[i][0]])
    vis.play(poses, interp='linear')
    time.sleep(5)
    vis.clear()

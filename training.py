from keras.utils import np_utils
import numpy as np
import h5py
from pointnet_model_class.pointnet_model import PointNet


def load_h5(h5_filename):
    """
    load the data from
    :param h5_filename: file path
    :return:
    """
    try:
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:]
        return data, label
    except Exception as ex:
        print(ex)


def rotate_point_data(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_data(batch_data, sigma=0.01, clip=0.05):
    """
    Randomly jitter points. jittering is per point.
    :param batch_data: BxNx3 array, original batch of point clouds
    :param sigma:
    :param clip:
    :return: BxNx3 point clouds
    """
    b, n, c = batch_data.shape
    jittered_data = np.clip(sigma * np.random.randn(b, n, c), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


train_path_file = "data/modelnet40_ply_hdf5_2048/train_files.txt"
with open(train_path_file, 'rt') as f:
    filenames = f.read().rstrip('\n').split('\n')


train_points = None
train_labels = None
for d in filenames:
    c_points, c_labels = load_h5(d)
    c_points = c_points.reshape(1, -1, 3)
    c_labels = c_labels.reshape(1, -1)
    if train_labels is None or train_points is None:
        train_labels = c_labels
        train_points = c_points
    else:
        train_labels = np.hstack((train_labels, c_labels))
        train_points = np.hstack((train_points, c_points))
train_points_reshape = train_points.reshape((-1, 2048, 3))
train_labels_reshape = train_labels.reshape(-1, 1)
train_points_reshape.astype(int)

# load test points and labels
test_path_file = "data/modelnet40_ply_hdf5_2048/test_files.txt"
with open(train_path_file, 'rt') as f:
    filenames = f.read().rstrip('\n').split('\n')

test_points = None
test_labels = None
for d in filenames:
    c_points, c_labels = load_h5(d)
    c_points = c_points.reshape(1, -1, 3)
    c_labels = c_labels.reshape(1, -1)
    if test_labels is None or test_points is None:
        test_labels = c_labels
        test_points = c_points
    else:
        test_labels = np.hstack((test_labels, c_labels))
        test_points = np.hstack((test_points, c_points))
test_points_reshape = test_points.reshape((-1, 2048, 3))
test_labels_reshape = test_labels.reshape(-1, 1)
test_labels_reshape.astype(int)

# label to categorical
Y_train = np_utils.to_categorical(train_labels_reshape, 40)
Y_test = np_utils.to_categorical(test_labels_reshape, 40)

model = PointNet().build()

# compile classification model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model on training data
for i in range(1, 60):
    train_points_rotate = rotate_point_data(train_points_reshape)
    train_points_jitter = jitter_point_data(train_points_rotate)
    model.fit(train_points_jitter, Y_train, batch_size=32, epochs=1, shuffle=True, verbose=1)
    print("Current epoch is:" + str(i))
    if i % 10 == 0:
        score = model.evaluate(test_points_reshape, Y_test, verbose=1)
        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])

# score the model
score = model.evaluate(test_points_reshape, Y_test, verbose=1)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

model.save("model/pointnet2048.h5")

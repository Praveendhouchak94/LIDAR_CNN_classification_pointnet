import pptk
from keras.models import load_model
import numpy as np
import tensorflow as tf
import time
import h5py


class DisplayPoint:
    def __init__(self):
        self.poses = None
        self.rotation_point()
        self.point_label = None
        self.point_data = None
        self.label = {
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
        self.model = None
        self.temp = True
        self.vis = None

    def rotation_point(self):
        """
        rotation of point cloud on pptk window
        :return:
        """
        self.poses = []
        self.poses.append([0, 0, 0, np.pi / 4, 0 * np.pi / 2, 5])
        self.poses.append([0, 0, 0, np.pi / 4, 1 * np.pi / 2, 5])
        self.poses.append([0, 0, 0, np.pi / 4, 2 * np.pi / 2, 5])
        self.poses.append([0, 0, 0, np.pi / 4, 3 * np.pi / 2, 5])
        self.poses.append([0, 0, 0, np.pi / 4, 4 * np.pi / 2, 5])

    @staticmethod
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

    def load_model(self, fname):
        """
        loading keras saved model
        :param fname: file path for keras model
        :return:
        """
        with tf.get_default_graph().as_default():
            self.model = load_model(fname, custom_objects={'tf': tf})

    def do_work(self, model_file, data_file, predict_signal):
        """
        started visualization loading data and model for point cloud data
        :param model_file: file path for keras model
        :param data_file: file path for point cloud data
        :param predict_signal: emit signal for predict and actual
        :return:
        """
        try:
            if self.point_label is None:
                self.load_data(data_file)
            if self.model is None:
                self.load_model(model_file)
            self.visualization_predict_point_cloud(predict_signal)
        except Exception as ex:
            print(ex)

    def visualization_predict_point_cloud(self, predict_signal):
        """
        open pptk window for visualization and predict
        :param predict_signal: signal to be emit to UI
        :return:
        """
        try:
            for i in range(self.point_label.shape[0]):
                point = np.expand_dims(self.point_data[i], axis=0)
                with tf.get_default_graph().as_default():
                    pred = self.model.predict(point)
                pred = np.argmax(pred)
                predict_signal.actual_predict.emit([self.label[self.point_label[i][0]], self.label[pred]])
                if self.temp:
                    self.vis = pptk.viewer(self.point_data[i])
                    self.vis.set(point_size=0.01, r=5, theta=1.574)
                    self.temp = False
                else:
                    self.vis.load(self.point_data[i])
                    self.vis.set(point_size=0.01, r=5, theta=1.574)
                self.vis.play(self.poses, interp='linear')
                time.sleep(3)
                self.vis.clear()
        except Exception as ex:
            print(ex)
        finally:
            self.temp = True
            self.vis.close()

    def load_data(self, fname):
        """
        loading point cloud data and reshaping
        :param fname: file path for point cloud data
        :return:
        """
        points, labels = self.load_h5(fname)
        self.point_data = points.reshape(-1, 2048, 3)
        self.point_label = labels.reshape(-1, 1)
        print("data loaded")

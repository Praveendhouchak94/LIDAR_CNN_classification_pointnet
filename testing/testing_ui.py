from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from testing.pptk_window import DisplayPoint
from PyQt5.QtCore import pyqtSignal, QObject
import threading
import sys


class Communicate(QObject):
    """
    class for emitting a signal for thread
    """
    actual_predict = pyqtSignal(list)


class UiMainWindow(QMainWindow):
    """
    main UI for application
    """

    def __init__(self):
        """
        init function of the class
        """
        super().__init__()
        self.setObjectName("MainWindow")
        self.resize(550, 100)
        style_load = """ QPushButton {
                    background-color: rgb(211,211,211);
                    color: black;
                    border-width: 2px;
                    border-color: #ae32a0;
                    border-style: solid;
                    font: bold 18px;
                    }
                    QPushButton:hover {
                    background-color: #64b5f6;
                    color: #fff;
                    }"""
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")

        self.button_load_logic = QtWidgets.QPushButton(self.centralwidget)
        self.button_load_logic.setText("Load Logic")
        self.button_load_logic.setStyleSheet(style_load)
        self.button_load_logic.clicked.connect(self.show_dialog_logic)

        self.button_load_data = QtWidgets.QPushButton(self.centralwidget)
        self.button_load_data.setText("Load data")
        self.button_load_data.setStyleSheet(style_load)
        self.button_load_data.clicked.connect(self.show_dialog_data)

        self.button_play = QtWidgets.QPushButton(self.centralwidget)
        self.button_play.setText("Play")
        self.button_play.setStyleSheet(""" QPushButton {
                    background-color: rgb(0,255,0);
                    color: black;
                    border-width: 2px;
                    border-color: #ae32a0;
                    border-style: solid;
                    font: bold 18px;
                    }
                    QPushButton:hover {
                    background-color: #64b5f6;
                    color: #fff;
                    }""")
        self.button_play.clicked.connect(self.show_point_cloud_data)

        self.button_Stop = QtWidgets.QPushButton(self.centralwidget)
        self.button_Stop.setText("Stop")
        self.button_Stop.setStyleSheet(""" QPushButton {
                    background-color: rgb(255,0,0);
                    color: black;
                    border-width: 2px;
                    border-color: #ae32a0;
                    border-style: solid;
                    font: bold 18px;
                    }
                    QPushButton:hover {
                    background-color: #64b5f6;
                    color: #fff;
                    }""")
        self.button_Stop.clicked.connect(self.close_point_cloud)

        self.gridLayout.addWidget(self.button_load_logic, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.button_load_data, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.button_play, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.button_Stop, 0, 3, 1, 1)

        self.label_actual = QtWidgets.QLabel(self.centralwidget)
        self.label_actual.setStyleSheet(""" QLabel {
                    background-color: rgb(211,211,211);
                    color: rgb(128,0,0);
                    font: bold 18px;}""")
        self.label_actual.setText("Actual 3D Object")

        self.label_predict = QtWidgets.QLabel(self.centralwidget)
        self.label_predict.setText("Predicted 3D Object")
        self.label_predict.setStyleSheet(""" QLabel {
                    background-color: rgb(211,211,211);
                    color: rgb(128,0,0);
                    font: bold 18px;}""")
        self.gridLayout.addWidget(self.label_actual, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.label_predict, 2, 0, 1, 1)

        self.label_actual_value = QtWidgets.QLabel(self.centralwidget)

        self.label_predict_value = QtWidgets.QLabel(self.centralwidget)

        self.gridLayout.addWidget(self.label_actual_value, 1, 1, 1, 3)
        self.gridLayout.addWidget(self.label_predict_value, 2, 1, 1, 3)

        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.setCentralWidget(self.centralwidget)

        self.data_file_path = None
        self.point_labels = None
        self.model_file_path = None
        self.actual_predict_class = Communicate()
        self.actual_predict_class.actual_predict.connect(self.display_result)
        self.display = DisplayPoint()

    def display_result(self, result):
        """
        display the actual and predict value in the UI
        :param result: list of actaul and predict value
        :return:
        """
        if result[0] == result[1]:
            self.label_actual_value.setStyleSheet("""QLabel {
                font: medium Ubuntu;
                font-size: 20px;
                Background-color: rgb(0,255,0);
                color: black;
                font-weight: bold;
            }""")
            self.label_predict_value.setStyleSheet("""QLabel {
                font: medium Ubuntu;
                font-size: 20px;
                Background-color: rgb(0,255,0);
                color: black;
                font-weight: bold;
            }""")

        else:
            self.label_actual_value.setStyleSheet("""QLabel {
                            font: medium Ubuntu;
                            font-size: 20px;
                            Background-color: rgb(255,0,0);
                            color: black;
                            font-weight: bold;
                        }""")
            self.label_predict_value.setStyleSheet("""QLabel {
                            font: medium Ubuntu;
                            font-size: 20px;
                            Background-color: rgb(255, 0, 0);
                            color: black;
                            font-weight: bold;
                        }""")
        self.label_actual_value.setText(result[0])
        self.label_predict_value.setText(result[1])

    def close_point_cloud(self):
        """
        close the pptk window
        :return:
        """
        self.display.vis.close()
        self.display.temp = True

    def show_dialog_data(self):
        """
        file dialog to load the data file
        :return:
        """
        filters = "Text files (*.txt);;Data h5 (*.h5)"
        selected_filter = "Data h5 (*.h5)"
        self.data_file_path = QFileDialog.getOpenFileName(self, 'load data file', '.', filters, selected_filter)[0]

    def show_dialog_logic(self):
        """
        file dialog to load the model file
        :return:
        """
        filters = "Text files (*.txt);;Model h5 (*.h5)"
        selected_filter = "Model h5 (*.h5)"
        self.model_file_path = QFileDialog.getOpenFileName(self, 'load logic file', '.', filters, selected_filter)[0]

    def show_point_cloud_data(self):
        """
        on play starting a thread for pptk window
        :return:
        """
        try:
            self.th = threading.Thread(target=self.display.do_work, args=(self.model_file_path,
                                                                              self.data_file_path,
                                                                              self.actual_predict_class))
            self.th.start()
        except Exception as ex:
            print(ex)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = UiMainWindow()
    ui.show()
    sys.exit(app.exec_())

# LIDAR_CNN_classification_pointnet
Deep Learning on Point cloud dataset for 3D Classification using keras library.
Orignal implementation of pointnet is done in tensorflow: https://github.com/charlesq34/pointnet .

currently I'm working on classification problem of 3D point cloud dataset after we month i will start working on Segmentation. 

Package used: keras, tensorflow, numpy, Pyqt5, pptk, h5py,threading

# classification
Download the raw point clouds dataset of <a href="http://modelnet.cs.princeton.edu/ModelNet40.zip" target="_blank">ModelNet40</a> models in .off files. It contain 40 class point cloud data that we want to classify using deep learning.
  
If you want to download the pointnet in HDF5 files that we have used <a href="https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip" target="_blank">data/modelNet40_ply_hdf5_2048</a>. Each point cloud contains 2048 points uniformly sampled from a point cloud data.

### Usage
To visualized point cloud data in h5py:

    cd point_cloud_hdf5_visualization
    python3 hdf5_file_visualization.py
   
To train the pointnet model in keras. I have train the model with google Colaboratory:

    python3 training.py
   
For visualization the output given by pointnet model

    cd testing
    python3 testing_ui.py
    
Your output look like this:

</p>

<div align="center"><img src ="https://github.com/Praveendhouchak94/LIDAR_CNN_classification_pointnet/blob/master/temp/output.png"  width="400" height="300" /></div>

Accuracy of the model is `83.7%`

References:

* https://colab.research.google.com/
* https://arxiv.org/abs/1612.00593
* https://modelnet.cs.princeton.edu/

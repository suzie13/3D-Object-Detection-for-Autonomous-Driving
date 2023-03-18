3D Object Detection using complex-yolov3 model architecture in Pytorch

DATASET used:
Kitti Dataset

https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d



HOW TO RUN ->

To train the dataset:
`python train.py`

To test the trained model weight on test images:
`python test.py`

STEPS: \

    1) Data Collection: LIDAR->(3D point clouds) and camera data->(2D images) from same scene. \
    2) Calibration: calibrate Lidar and camera data by determining relative positions and orientations \
       of LIDAR sensor and camera sensor so that can be accurately combined. [transformation]. \ 
       [mapping points from Lidar coordinate system to camera coordinate system]
    3) Point Cloud Processing: processing, discretization, remove noise/outliers, unique points, sorting. \
    4) Lidar to camera transformation: Once calibrated, the LIDAR points can be projected onto  \
    the camera image plane. Transforming LIDAR points from Lidar coordinate system to camera \
    coordinate system using transformation matrix we got from calibration. The resulting points \
    in camera coordinate system then projected on camera image plane using the camera parameters. \
    5) Object detection part: This projected LIDAR points on camera image plane is used to detect objects. \
    Neural network to the camera image. -> to get/predict location and class of each object.
    6) 3D bounding box estimation: Once object 2D detection done, their 3D bounding boxes estimated \
    from LIDAR point cloud. Finding the LIDAR points that correspond to each object in camera image \
    and from that -> estimating object’s position, orientation, and size in 3D space.
    7) Camera to LIDAR box transformation: Finally, this 3D bounding box in camera coordinate system \
     can be transformed back to LIDAR coordinate system using inverse of the transformation matrix \
     we got from calibration. Here we can represent in LIDAR point cloud system additionally.
    8) Plotting: For visualizing the results [Matplotlib, Mayavi]
    Estimated from before position, orientation, and size. 3D frame of the box overlay on LIDAR \
    point cloud. To get the location and size of the object relative to the surrounding. To get 3D \
    boxes on camera images this projected back to camera image plane using the camera parameters, \
    along with the 2D detections. \

References:

https://doi.org/10.48550/arXiv.2004.10934
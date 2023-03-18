3D Object Detection using complex-yolov3 model architecture in Pytorch


3D Object-Detection 

![front_tile_3D](https://user-images.githubusercontent.com/39700209/226108691-ea236f73-bea4-4852-ac90-24cb61b7f62b.gif)





Camera RGB image(left) and Lidar 2D image(right) side to side

![side_to_side2](https://user-images.githubusercontent.com/39700209/226110047-b811844b-82dd-407e-bd62-10b65fef1cd7.png)

![side_to_side3](https://user-images.githubusercontent.com/39700209/226110060-75691c0e-ddf4-4553-bd79-30618bcddc2f.png)

![side_to_side4](https://user-images.githubusercontent.com/39700209/226110067-5f1547a6-1383-4db3-bd20-35c51bf393d5.png)

![side_to_side5](https://user-images.githubusercontent.com/39700209/226110075-76b537fb-0112-4633-a36f-fba3d5c2bc83.png)


The yellow bounding box on the 2D LIDAR image represents the Class of Car (index 0),
red bounding box represents Pedestrian (index 1),
blue bounding box represents Cyclist (index 2)

The different colored (light blue) edge of the bounding box in 2D Lidar image represents 
the front of the detected objected which is also the likely direction of the detected object to
move in.


DATASET used:

Kitti Dataset
The Kitti dataset consists of:

    calib folder: This contains the caliberation files for cmaera and velodyne sensor.

    image_2 folder: contains rgb images from left camera.

    label_2 folder: contains ground truth annotations for detection and tracking.

    velodyne folder: contains the velodyne LIDAR point cloud in bin files.



Each file of label_2 has 15 columns ->

    column 0 = classes (car, pedestrian, van, truck, cycle, tram, person sitting, misc, don't care)

    column 1 = Truncation (float value from 0 to 1 where 0 is fully visible and 1 is fully truncated 
                outside the image boundaries)(how much that object is visible in the image)

    column 2 = Occlussion (0 to 3 where 0 is not occluded and 3 is fully occluded by other objects 
                in the scene)

    column 3 = Alpha (observation angle in radians from center of camera)

    column 4,5,6,7 = bounding box (2D bounding box coordinate i.e. x, y, w, h)

    column 8,9,10 = Dimensions (3D object dimensions i.e height, width, length (in meteres))

    column 11,12,13 = The center location x, y, z of the 3D object in camera coordinates (in meteres)

    column 14 = Rotation (ry) around Y-axis (vertical axis) in camera coordinates [-pi ..pi]


https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d



HOW TO RUN ->

To train the dataset:
`python train.py`

To test the trained model weight on test images:
`python test.py`

STEPS: 

    1) Data Collection: LIDAR->(3D point clouds) and camera data->(2D images) from same scene. 

    2) Calibration: calibrate Lidar and camera data by determining relative positions and 
    orientations of LIDAR sensor and camera sensor so that can be accurately combined. 
    [transformation]. [mapping points from Lidar coordinate system to camera coordinate system]

    3) Point Cloud Processing: processing, discretization, remove noise/outliers, unique points, 
        sorting. 

    4) Lidar to camera transformation: Once calibrated, the LIDAR points can be projected onto  
        the camera image plane. Transforming LIDAR points from Lidar coordinate system to camera 
        coordinate system using transformation matrix we got from calibration. The resulting points 
        in camera coordinate system then projected on camera image plane using the camera parameters. 

    5) Object detection part: This projected LIDAR points on camera image plane is used to detect 
        objects. Neural network to the camera image. To predict location and class of each object.

    6) 3D bounding box estimation: Once object 2D detection done, their 3D bounding boxes estimated 
        from LIDAR point cloud. Finding the LIDAR points that correspond to each object in camera 
        image and from that estimating object’s position, orientation, and size in 3D space.

    7) Camera to LIDAR box transformation: Finally, this 3D bounding box in camera coordinate system 
        can be transformed back to LIDAR coordinate system using inverse of the transformation 
        matrix we got from calibration. Here we can represent in LIDAR point cloud system 
        additionally.

    8) Plotting: For visualizing the results [Matplotlib, Mayavi]
        Estimated from before position, orientation, and size. 3D frame of the box overlay on LIDAR 
        point cloud. To get the location and size of the object relative to the surrounding. To get 
        3D boxes on camera images this projected back to camera image plane using the camera 
        parameters, along with the 2D detections. 

References:

https://doi.org/10.48550/arXiv.2004.10934
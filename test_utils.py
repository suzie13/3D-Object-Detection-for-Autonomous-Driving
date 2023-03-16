import numpy as np
import mayavi.mlab as mlab
import cv2
from kitti_utils import *
from utils import *


def draw_lidar(pc, color=None, fig1=None, bgcolor=(0,0,0), pts_scale=1, pts_mode='point', pts_color=None):
    x_min = 0
    x_max = 40
    y_min = -20
    y_max = 20

    mlab.clf(figure=None)
    if color is None: color = pc[:,2]
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=pts_color, mode=pts_mode, colormap = 'gnuplot', scale_factor=pts_scale, figure=fig1)
    
    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
    
    #draw axis
    axes=np.array([[2.,0.,0.,0.],
                   [0.,2.,0.,0.],
                   [0.,0.,2.,0.],],dtype=np.float64)
	
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig1)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig1)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig1)

    fov=np.array([[20., 20., 0.,0.],[20.,-20., 0.,0.],],dtype=np.float64)
    
    mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig1)
    mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig1)
    

    mlab.plot3d([x_min, x_min], [y_min, y_max], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig1)
    mlab.plot3d([x_max, x_max], [y_min, y_max], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig1)
    mlab.plot3d([x_min, x_max], [y_min, y_max], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig1)
    mlab.plot3d([x_min, x_max], [y_max, y_max], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig1)
    
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=60.0, figure=fig1)
    return fig1




def show_image_with_boxes(img, objects, calib, show3d=False):
    ''' Show image with 2D bounding boxes '''
    image_2 = np.copy(img) # for 3d bbox
    for obj in objects:
        if obj.type=='DontCare':continue

        box3d_pts_2d, _ = compute_box_3d(obj, calib.P)
        if box3d_pts_2d is not None:

            box3d_pts_2d = box3d_pts_2d.astype(np.int32)
            for k in range(0,4):
                i,j = k,(k+1) % 4
                cv2.line(image_2, (box3d_pts_2d[i,0],box3d_pts_2d[i,1]), (box3d_pts_2d[j,0],box3d_pts_2d[j,1]), (255,0,255), 2)

                i,j = k+4,(k+1) % 4 + 4
                cv2.line(image_2, (box3d_pts_2d[i,0],box3d_pts_2d[i,1]), (box3d_pts_2d[j,0],box3d_pts_2d[j,1]), (255,0,255), 2)

                i,j=k,k+4
                cv2.line(image_2, (box3d_pts_2d[i,0],box3d_pts_2d[i,1]), (box3d_pts_2d[j,0],box3d_pts_2d[j,1]), (255,0,255), 2)
    if show3d:
        cv2.imshow("img", image_2)
    return image_2

def show_lidar_with_boxes(pc_velo, objects, calib, img_fov=False, img_width=None, img_height=None, fig=None): 
    if not fig:
        fig = mlab.figure(figure="Point Cloud", bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1250, 550))

    if img_fov:
        points_2D = calib.project_velo_to_image(pc_velo)
        points_fieldofview = (points_2D[:,0]<img_width) & (points_2D[:,0]>=0) & (points_2D[:,1]<img_height) & (points_2D[:,1] >= 0) & (pc_velo[:,0] > 0.0)
        pc_velo = pc_velo[points_fieldofview,:]

    draw_lidar(pc_velo, fig1=fig)

    for obj in objects:

        if obj.type=='DontCare':
            continue
        # Draw 3d bounding box
        __, box3d_pts_3d = compute_box_3d(obj, calib.P) 
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)

        # heading arrow
        _, ori3d_pts_3d = compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1,y1,z1 = ori3d_pts_3d_velo[0,:]
        x2,y2,z2 = ori3d_pts_3d_velo[1,:]

        for n in range(len([box3d_pts_3d_velo])):
            b = ([box3d_pts_3d_velo])[n]
            mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%n, scale=(1,1,1), color=(0,1,1), figure=fig)
            for k in range(0,4):
                i,j=k,(k+1)%4
                mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=(1,1,1), tube_radius=None, line_width=2, figure=fig)

                i,j=k+4,(k+1)%4 + 4
                mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=(1,1,1), tube_radius=None, line_width=2, figure=fig)

                i,j=k,k+4
                mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=(1,1,1), tube_radius=None, line_width=2, figure=fig)
        
        mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    mlab.view(distance=90)


def yolo_labels_inv(targets, bc):
    num_t = 0
    for i, t in enumerate(targets):
        if t.sum(0):
            num_t += 1
    
    labels = np.zeros([num_t, 8], dtype=np.float32)
    n = 0
    for t in targets:
        if t.sum(0) == 0:
            continue
        c, y, x, w, l, im, re = t        
        z, h = -1.55, 1.5
        if c == 1: 
            h = 1.8
        elif c == 2:
            h = 1.4
            
        y = y * (bc["maxY"] - bc["minY"]) + bc["minY"]
        x = x * (bc["maxX"] - bc["minX"]) + bc["minX"]
        w = w * (bc["maxY"] - bc["minY"])
        l = l * (bc["maxX"] - bc["minX"])

        w -= 0.3
        l -= 0.3

        labels[n, :] = c, x, y, z, h, w, l, - np.arctan2(im, re) - 2*np.pi
        n += 1

    return labels


def drawRotatedBox(img,x,y,w,l,yaw,color):
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2)
    print(corners_int)
    print(corners_int.shape)
    print(".....")
    print(corners_int[0, 0])
    print(corners_int[0, 1])
    print(corners_int[3, 0])
    print(corners_int[3, 1])
    cv2.line(img, (int(corners_int[0, 0]), int(corners_int[0, 1])), (int(corners_int[3, 0]), int(corners_int[3, 1])), (255, 255, 0), 2)

def draw_box_in_bev(rgb_map, target):
    for j in range(50):
        if(np.sum(target[j,1:]) == 0):continue
        cls_id = int(target[j][0])
        x = target[j][1] * 608
        y = target[j][2] * 608
        w = target[j][3] * 608
        l = target[j][4] * 608
        yaw = np.arctan2(target[j][5], target[j][6])
        drawRotatedBox(rgb_map, x, y, w, l, yaw, colors[cls_id])

def rescale_boxes(bbox, curr_dim, initial_shape):
    init_h, init_w = initial_shape
    #padding values
    pad_x = max(init_h - init_w, 0) * (curr_dim / max(initial_shape))
    pad_y = max(init_w - init_h, 0) * (curr_dim / max(initial_shape))
    # new Image height and width
    h = curr_dim - pad_y
    w = curr_dim - pad_x
    # Rescale bbox to initial image shape
    bbox[:, 0] = ((bbox[:, 0] - pad_x // 2) / w) * init_w
    bbox[:, 1] = ((bbox[:, 1] - pad_y // 2) / h) * init_h
    bbox[:, 2] = ((bbox[:, 2] - pad_x // 2) / w) * init_w
    bbox[:, 3] = ((bbox[:, 3] - pad_y // 2) / h) * init_h

    return bbox



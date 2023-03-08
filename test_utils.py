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

    f_v=np.array([[20., 20., 0.,0.],[20.,-20., 0.,0.],],dtype=np.float64)
    
    mlab.plot3d([0, f_v[0,0]], [0, f_v[0,1]], [0, f_v[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig1)
    mlab.plot3d([0, f_v[1,0]], [0, f_v[1,1]], [0, f_v[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig1)
    

    mlab.plot3d([x_min, x_min], [y_min, y_max], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig1)
    mlab.plot3d([x_max, x_max], [y_min, y_max], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig1)
    mlab.plot3d([x_min, x_max], [y_min, y_max], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig1)
    mlab.plot3d([x_min, x_max], [y_max, y_max], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig1)
    
    #mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=60.0, figure=fig1)
    return fig1
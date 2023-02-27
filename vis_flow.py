import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_flow(image, flow_image, confidence, threshmin=10):
    """
    params:
        @img: np.array(h, w)
        @flow_image: np.array(h, w, 2)
        @confidence: np.array(h, w)
        @threshmin: confidence must be greater than threshmin to be kept
    return value:
        None
    """

    """
    STUDENT CODE BEGINS
    """
    fx=[]
    fy=[]
    x=[]
    y=[]

    for i in range(confidence.shape[1]):
        for j in range(confidence.shape[0]):
            if (confidence[j,i]>threshmin):
                fx.append([flow_image[j,i,0]])
                x.append(i)
                fy.append([flow_image[j,i,1]])
                y.append(j)
                
    
    fx = np.array(fx)
    fy = np.array(fy)
    x = np.array(x)
    y = np.array(y)

    flow_x = fx
    
    """
    STUDENT CODE ENDS
    """
    
    plt.imshow(image, cmap='gray')
    plt.quiver(x, y, (flow_x*10).astype(int), (fy*10).astype(int), 
                    angles='xy', scale_units='xy', scale=1., color='red', width=0.001)
    
    return





    


import numpy as np
import pdb

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
        @x: int
        @y: int
    return value:
        flow: np.array(2,)
        conf: np.array(1,)
    """
    """
    STUDENT CODE BEGINS
    """
    fx = Ix
    fy = Iy
    ft = It
    j = x
    i = y

    #Fx = fx[(i - 2):(i + 1 + 2), (j - 2):(j + 1 + 2)]
    #Fy = fy[(i - 2):(i + 1 + 2), (j - 2):(j + 1 + 2)]
    #Ft = -ft[(i - 2):(i + 1 + 2), (j - 2):(j + 1 + 2)]

    Fx = fx[max(0,(i - size//2)):(i + 1 + size//2), max(0,(j - size//2)):(j + 1 + size//2)]
    Fy = fy[max(0,(i - size//2)):(i + 1 + size//2), max(0,(j - size//2)):(j + 1 + size//2)]
    Ft = -ft[max(0,(i - size//2)):(i + 1 + size//2), max(0,(j - size//2)):(j + 1 + size//2)]


    Fx = np.transpose(Fx)
    Fy = np.transpose(Fy)
    Ft = np.transpose(Ft)

    Fx = Fx.flatten()
    Fy = Fy.flatten()
    Ft = Ft.flatten()

    #print(np.shape(Fx))

    A = np.transpose([Fx, Fy])
    #print(np.shape(A))

    # Least square fit to solve for unknown
    Uall = np.linalg.lstsq(A, Ft, rcond=-1)
    U = Uall[0]

    # 2x1 matrix containing the velocity vector for a single pixel point
    u = U[0]
    v = U[1]

    flow = [[u],[v]]
    flow = np.reshape(flow, (2,))

    if len(Uall[3]) == 0:
      conf = 0
    else:
      conf = Uall[3][1]
 
    """
    STUDENT CODE ENDS
    """
    return flow, conf


def flow_lk(Ix, Iy, It, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
    return value:
        flow: np.array(h, w, 2)
        conf: np.array(h, w)
    """
    image_flow = np.zeros([Ix.shape[0], Ix.shape[1], 2])
    confidence = np.zeros([Ix.shape[0], Ix.shape[1]])
    for x in range(Ix.shape[1]):
        for y in range(Ix.shape[0]):
            flow, conf = flow_lk_patch(Ix, Iy, It, x, y)
            image_flow[y, x, :] = flow
            confidence[y, x] = conf
    return image_flow, confidence

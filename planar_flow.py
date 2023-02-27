import numpy as np

def compute_planar_params(flow_x, flow_y, K,
                                up=[256, 0], down=[512, 256]):
    """
    params:
        @flow_x: np.array(h, w)
        @flow_y: np.array(h, w)
        @K: np.array(3, 3)
        @up: upper left index [i,j] of image region to consider.
        @down: lower right index [i,j] of image region to consider.
    return value:
        sol: np.array(8,)
    """
    """
    STUDENT CODE BEGINS
    """

    
    K_norm = K/K[2,2]
    fx = K_norm[0,0]
    u = flow_x/fx
    fy = K_norm[1,1]
    v = flow_y/fy
    Kinv = np.linalg.inv(K)

    A=[]
    B=[]
    upy = up[0]
    upx = up[1]
    downy = down[0]
    downx = down[1]

    for i in range(upy,downy):
        for j in range(upx,downx):
            
            B.append([u[i,j]])
            B.append([v[i,j]])
            X = np.array([j,i,1])
            X_inverse = np.matmul(Kinv, X)
            y = X_inverse[1]/X_inverse[2]
            x = X_inverse[0]/X_inverse[2]
            
            A.append([x**2, x*y, x, y, 1, 0, 0, 0])
            A.append([x*y, y**2, 0, 0, 0, y, x, 1])
            
    
    solution = np.linalg.lstsq(np.array(A), np.array(B), rcond=None)[0]
    sol = solution.flatten()
    """
    STUDENT CODE ENDS
    """

    return sol
    

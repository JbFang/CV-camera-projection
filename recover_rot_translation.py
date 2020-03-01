import numpy as np
from scipy.spatial.transform import Rotation

def recover_E_from_F(f_matrix, k_matrix):
    '''
    Recover the essential matrix from the fundamental matrix

    Args:
    -   f_matrix: fundamental matrix as a numpy array
    -   k_matrix: the intrinsic matrix shared between the two cameras
    Returns:
    -   e_matrix: the essential matrix as a numpy array (shape=(3,3))
    '''

    e_matrix = None

    ##############################
    # TODO: Student code goes here
    e_matrix = np.dot(np.dot(k_matrix.T,f_matrix), k_matrix)
    

    # raise NotImplementedError
    ##############################

    return e_matrix

def recover_rot_translation_from_E(e_matrix):
    '''
    Decompose the essential matrix to get rotation and translation (upto a scale)

    Ref: Section 9.6.2 

    Args:
    -   e_matrix: the essential matrix as a numpy array
    Returns:
    -   R1: the 3x1 array containing the rotation angles in radians; one of the two possible
    -   R2: the 3x1 array containing the rotation angles in radians; other of the two possible
    -   t: a 3x1 translation matrix with unit norm and +ve x-coordinate; if x-coordinate is zero then y should be positive, and so on.

    '''

    R1 = None
    R2 = None
    t = None

    ##############################
    # TODO: Student code goes here
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    Z = np.array([[0,1,0],[-1,0,0],[0,0,0]])
    
    
    u, s, v = np.linalg.svd(e_matrix)
    r1 = Rotation.from_dcm(np.dot(np.dot(u,W), v))
    r2 = Rotation.from_dcm(np.dot(np.dot(u,W.T), v))

    R1 = r1.as_rotvec()
    R2 = r2.as_rotvec()

    t = u[:,-1]
    if t[0] < 0:
        t = -t
    elif t[0] == 0 and t[1] < 0:
        t = -t
    elif t[1] == 0 and t[2] < 0:
        t = -t

    # raise NotImplementedError
    ##############################

    return R1, R2, t

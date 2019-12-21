import math


def calculate_angle(x, y, z):
    v1 = [y[0] - x[0], y[1] - x[1]]
    v2 = [z[0] - x[0], z[1] - x[1]]
    dot_prod = v1[0]*v2[0] + v1[1]*v2[1]
    v1_length = math.sqrt(v1[0]**2 + v1[1]**2)
    v2_length = math.sqrt(v2[0]**2 + v2[1]**2)
    cos_theta = dot_prod/(v1_length * v2_length)
    theta = math.acos(cos_theta)
    return theta


def classify_traj(trajectory: list):
    '''
    :param trajectory: a list of (x,y) location points represent trajectory
    :return: the category of this trajectory, curve, straight or change lane
    '''

    traj_len = len(trajectory)
    if traj_len < 50:
        return "invalid"
    curvature_threshold = 0.03
    lane_change_threshold = 0.07
    valid_threshold = 0.1

    x = trajectory[0]
    y = trajectory[1]
    z = trajectory[-1]

    angle = calculate_angle(x, y, z)

    if angle < curvature_threshold:
        return "straight"
    elif angle < lane_change_threshold:
        return "curve"
    elif angle < valid_threshold:
        return "changeLane"
    else:
        return "invalid"


def safetyindex(x):
    '''
    :param x: state for ego vihicle
    :return: safety index
    '''
    assert x.shape[1] == 66
    d_min = 10#50
    d = x[0][63]
    # k_v=2
    k_v = 1
    yita=10
    dot_d = x[0][64] 
    lambd = 0.5 # uncertainty margin
    dT = .1
    # phi = d_min**2 + yita * dT + lambd * dT - d**2 - k_v * dot_d
    phi = d_min**2 - d**2 - k_v*dot_d
    return phi
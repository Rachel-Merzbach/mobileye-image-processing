import numpy as np
import math


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)

    if abs(tZ) < 10e-6:
        print('tz = ', tZ)
    elif norm_prev_pts.size == 0:
        print('no prev points')
    elif norm_prev_pts.size == 0:
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)

    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))

    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []

    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        #
        # if not valid:
        #     Z = 0
        if valid:
            validVec.append(valid)
            P = Z * np.array([p_curr[0], p_curr[1], 1])
            pts_3D.append((P[0], P[1], P[2]))
            corresponding_ind.append(corresponding_p_ind)

    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    pts = np.array(pts)
    # transform pixels into normalized pixels using the focal length and principle point
    x = (pts[:, 0] - pp[0]) / focal
    y = (pts[:, 1] - pp[1]) / focal
    return np.column_stack((np.column_stack((x, y)), np.array([1] * len(pts), dtype=float)))


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    x = np.array(pts[:, 0] * focal + pp[0])
    y = np.array(pts[:, 1] * focal + pp[1])
    return np.column_stack((np.column_stack((x, y)), np.array([1] * len(pts), dtype=float)))


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3, :3]
    T = EM[:3, -1]
    foe = (T[0] / T[2], T[1] / T[2])
    return R, foe, T[2]


def rotate(pts, R):
    # rotate the points using R
    res = []
    for pt in pts:
        v = R.dot(pt)
        res.append([v[0] / v[2], v[1] / v[2], 1])
    return np.array(res)


def find_corresponding_points(p, norm_pts_rot, foe):
    distance = []
    distance_indexes = {}
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0] - p[0] * foe[1]) / (foe[0] - p[0])

    # run over all norm_pts_rot and find the one closest to the epipolar line
    for index, coordinate in enumerate(norm_pts_rot):
        distance.append(abs((m * coordinate[0] - coordinate[1] + n) / math.sqrt(m ** 2 + 1)))
        distance_indexes.update({index: coordinate})

    # find the closest point and return it and its index
    min_dist_idx = distance.index(min(distance))
    return min_dist_idx, distance_indexes[min_dist_idx]


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance between foe[x] and p_curr[y], foe[y] and p_curr[y]
    diff_x = abs(foe[0] - p_curr[0])
    diff_y = abs(foe[1] - p_curr[1])

    # calculate the estimated x and y
    estimated_x = tZ * (foe[0] - p_rot[0]) / (p_curr[0] - p_rot[0])
    estimated_y = tZ * (foe[1] - p_rot[1]) / (p_curr[1] - p_rot[1])

    # calculate the ratio
    ratio = diff_x / (diff_y + diff_x)

    # calculate the estimated z
    return estimated_x * ratio + estimated_y * (1 - ratio)

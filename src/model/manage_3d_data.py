import numpy as np
import math

class Manage3dData:
    def __init__(self, prev_container, curr_container, focal, pp):
        self.focal = focal
        self.pp = pp
        self.curr_container = curr_container
        self.norm_prev_pts = self.normalize(prev_container.traffic_light)
        self.norm_curr_pts = self.normalize(curr_container.traffic_light)
        self.R, self.foe, self.tZ = Manage3dData.decompose(np.array(self.curr_container.EM))

    def calc_TFL_dist(self):
        if self.norm_prev_pts.size == 0:
            print('no prev points')
        elif self.norm_curr_pts.size == 0:
            print('no curr points')
        else:
            self.curr_container.corresponding_ind, self.curr_container.tfls_3d_location, self.curr_container.valid = self.calc_3D_data()

        return self.curr_container

    def calc_3D_data(self):
        norm_rot_pts = self.rotate(self.norm_prev_pts)
        pts_3D = []
        corresponding_ind = []
        valid_vec = []

        for p_curr in self.norm_curr_pts:
            corresponding_p_ind, corresponding_p_rot = self.find_corresponding_points(p_curr, norm_rot_pts)
            Z = self.calc_dist(p_curr, corresponding_p_rot)
            valid = (Z >= 1)
            if valid:
                valid_vec.append(valid)
                P = Z * np.array([p_curr[0], p_curr[1], 1])
                pts_3D.append((P[0], P[1], P[2]))
                corresponding_ind.append(corresponding_p_ind)

        return corresponding_ind, np.array(pts_3D), valid_vec

    def normalize(self, pts):
        pts = np.array(pts)
        # transform pixels into normalized pixels using the focal length and principle point
        x = (pts[:, 0] - self.pp[0]) / self.focal
        y = (pts[:, 1] - self.pp[1]) / self.focal
        return np.column_stack((np.column_stack((x, y)), np.array([1] * len(pts), dtype=float)))

    def unnormalize(self, pts):
        pts = np.array(pts)
        # transform normalized pixels into pixels using the focal length and principle point
        x = np.array(pts[:, 0] * self.focal + self.pp[0])
        y = np.array(pts[:, 1] * self.focal + self.pp[1])
        return np.column_stack((np.column_stack((x, y)), np.array([1] * len(pts), dtype=float)))

    @staticmethod
    def decompose(EM):
        # extract R, foe and tZ from the Ego Motion
        R = EM[:3, :3]
        T = EM[:3, -1]
        foe = (T[0] / T[2], T[1] / T[2])
        return R, foe, T[2]

    def rotate(self, pts):
        # rotate the points using R
        res = []
        for pt in pts:
            v = self.R.dot(pt)
            res.append([v[0] / v[2], v[1] / v[2], 1])
        return np.array(res)

    def find_corresponding_points(self, p, norm_pts_rot):
        distance = []
        distance_indexes = {}
        m = (self.foe[1] - p[1]) / (self.foe[0] - p[0])
        n = (p[1] * self.foe[0] - p[0] * self.foe[1]) / (self.foe[0] - p[0])

        # run over all norm_pts_rot and find the one closest to the epipolar line
        for index, coordinate in enumerate(norm_pts_rot):
            distance.append(abs((m * coordinate[0] - coordinate[1] + n) / math.sqrt(m ** 2 + 1)))
            distance_indexes.update({index: coordinate})

        # find the closest point and return it and its index
        min_dist_idx = distance.index(min(distance))
        return min_dist_idx, distance_indexes[min_dist_idx]

    def calc_dist(self, p_curr, p_rot):
        # calculate the estimated x and y
        estimated_x = self.tZ * (self.foe[0] - p_rot[0]) / (p_curr[0] - p_rot[0])
        estimated_y = self.tZ * (self.foe[1] - p_rot[1]) / (p_curr[1] - p_rot[1])

        # calculate the distance between foe[x] and p_curr[x], foe[y] and p_curr[y]
        diff_x = abs(self.foe[0] - p_curr[0])
        diff_y = abs(self.foe[1] - p_curr[1])

        # calculate the ratio
        ratio = diff_x / (diff_y + diff_x)

        # calculate the estimated z
        return estimated_x * ratio + estimated_y * (1 - ratio)

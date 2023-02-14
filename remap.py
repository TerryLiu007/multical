import pickle
from multical import tables
import numpy as np
import json
import cv2
np.set_printoptions(precision=4, suppress=True)

def to_json_list(content):
    for name, arr in content.items():
        if isinstance(arr, np.ndarray):
            content[name] = arr.reshape(-1).tolist()
        elif isinstance(arr, dict):
            to_json_list(arr)


class ZEO_object:
    def __init__(self, intrinsic, extrinsic, homo):
        self.parameters = []
        for i, item in enumerate(intrinsic):
            triang = {}
            triang["K"] = item.intrinsic
            triang["Ko"] = item.intrinsic
            triang["distCoeff"] = item.dist
            triang["R"] = extrinsic[i][:3, :3]
            triang["T"] = extrinsic[i][:3, 3]
            triang["H"] = homo[i]

            image_size = item.image_size
            triang["imgSize"] = image_size
            triang["rectifyAlpha"] = 0.0
            roi = {}
            roi["height"] = image_size[1]
            roi["width"] = image_size[0]
            roi["x"] = 0
            roi["y"] = 0
            triang["validPixROI"] = roi

            self.parameters.append(triang)


class Checkerboard:
    def __init__(self, pattern, size):
        self.matrix = np.zeros((pattern[0]*pattern[1], 3))
        for i in range(pattern[1]):
            for j in range(pattern[0]):
                self.matrix[i * pattern[0] + j, 0] = (i+1) * size
                self.matrix[i * pattern[0] + j, 1] = (j+1) * size


def undistort_points(info, points):
    undistorted = cv2.undistortPoints(points.reshape(-1, 1, 2), info.intrinsic, info.dist, P=info.intrinsic)
    return undistorted.reshape(*points.shape[:-1], 2)


f = open('calibration.pkl', 'rb')
data = pickle.load(f)

det = data.point_table.points
ground = det[:, 0, 0, :, :]
reprojection_error = data.calibrations['calibration'].reprojection_error[:ground.shape[0]*ground.shape[1]]
print('Reprojection error in average is: {}'.format(reprojection_error.mean()))

calib = data.calibrations['calibration']
# moving cameras
# view_poses = tables.inverse(tables.expand_views(calib.pose_estimates))
# camera_poses = view_poses._index[:, 0]
# board_poses = calib.pose_estimates.board
# moving boards
# camera_poses = tables.inverse(calib.pose_estimates.camera)
# board_poses = tables.expand_boards(calib.pose_estimates)._index[0]

est = data.calibrations['calibration'].pose_estimates
camera_poses = tables.expand_views(est)._index[:, 0].poses
board_poses = est.board.poses

anchor = board_poses[0]
new_camera_poses = []

for cam_pose in camera_poses:
    # inv_anchor = np.linalg.inv(anchor)
    # trans = np.matmul(cam_pose, inv_anchor)
    trans = cam_pose
    new_camera_poses.append(trans)
intrinsics = data.calibrations['calibration'].cameras.param_objects

homographs = []
cb = Checkerboard((8, 11), 60)
board_points = data.calibrations['calibration'].board_points.points
points_3d = cb.matrix
for cam_ind, points_2d in enumerate(ground):
    points_2d = undistort_points(intrinsics[cam_ind], points_2d)
    homograph = cv2.findHomography(points_2d, points_3d)
    homographs.append(homograph[0])

zeo_object = ZEO_object(intrinsics, new_camera_poses, homographs)
for i in range(len(zeo_object.parameters)):
    to_json_list(zeo_object.parameters[i])
with open('sdk_parameters.json', 'w') as f:
    json.dump(zeo_object.parameters, f, indent=4)

print('done')
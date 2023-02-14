import os.path as osp
import os
import cv2
import json
import numpy as np
import glob

data_folder = '20230208'
fix_cam = [i for i in range(6)]

image_load_dir = osp.join('..', 'data', data_folder)
extrinsic_load_path = 'sdk_parameters.json'
output_path = osp.join('..', 'data', data_folder+'_homo_check')

if not osp.isdir(output_path):
    os.makedirs(output_path)


def load_images(load_dir):
    """
    Load all images from a directory
    :param load_dir: directory where images are
    :return: list of cv2 images, and pathname stems
    """
    image_pathnames = sorted(glob.glob(osp.join(load_dir, '*')))
    images = []
    for image_pathname in [image_pathnames[0]]:
        try:
            assert osp.isfile(image_pathname)
            image = cv2.imread(image_pathname)
            images.append(image)

        except Exception:
            print('Fail to load', image_pathname)

    return images


def undistort_one(img, mtx, dist, new_mtx, roi):
    """
    Perform undistortion with parameters
    :param img: source image
    :param camera_matrix:
    :param distortion_coefficients:
    :param new_camera_matrix:
    :param roi: region of interest
    :return:
    """
    mtx = np.array(mtx).reshape(3, 3)
    dist = np.array(dist)
    new_mtx = np.array(new_mtx).reshape(3, 3)
    undist = cv2.undistort(img, mtx, dist, None, new_mtx)
    x = roi['x']
    y = roi['y']
    w = roi['width']
    h = roi['height']
    cropped = undist[y:y + h, x:x + w]

    return cropped


def homo_transform(image, ref, homo_mat):
    homo_mat = np.array(homo_mat).reshape(3, 3)
    im_dst = cv2.warpPerspective(image, homo_mat, ref.shape[:2])
    return cv2.addWeighted(np.transpose(im_dst, (1, 0, 2)), 0.5, ref, 0.5, 0)


class Checkerboard:
    def __init__(self, pattern, size, mode):
        if mode == 'full':
            self.matrix = np.zeros((pattern[0]*pattern[1], 3))
            for i in range(pattern[1]):
                for j in range(pattern[0]):
                    self.matrix[i * pattern[0] + j, 0] = i * size
                    self.matrix[i * pattern[0] + j, 1] = j * size
        if mode == 'x':
            self.matrix = np.zeros((pattern[1], 3))
            for i in range(pattern[1]):
                self.matrix[i, 0] = i * size
        if mode == 'y':
            self.matrix = np.zeros((pattern[0], 3))
            for i in range(pattern[0]):
                self.matrix[i, 1] = i * size

def main():
    # for each camera
    reference = cv2.imread(osp.join('..', 'data', 'ref.png'))
    f = open(extrinsic_load_path, 'r')
    extrinsics = json.load(f)

    cb = Checkerboard((8, 11), 6, 'y')
    points_3d = cb.matrix / 100

    for cam_idx in fix_cam:
        extrinsic = extrinsics[cam_idx]
        homo_mat = extrinsic['H']
        images = load_images(osp.join(image_load_dir, 'cam{0:01d}'.format(cam_idx+1)))

        # compute homography transformed image
        for image in images:
            R = np.array(extrinsic['R']).reshape(3, 3)
            T = np.array(extrinsic['T']).reshape(3, 1)
            K = np.array(extrinsic['K']).reshape(3, 3)
            distCoeff = np.array(extrinsic['distCoeff'])
            projected, _ = cv2.projectPoints(points_3d.reshape(-1, 1, 3), R, T, K, distCoeff)

            for pts in projected:
                if 0 <= pts[0,0] < image.shape[0] and 0 <= pts[0,1] < image.shape[1]:
                    y = int(pts[0, 0])
                    x = int(pts[0, 1])
                    image[x-2:x+2, y-2:y+2] = [0, 0, 255]
            cv2.imwrite('{}_ex.png'.format(osp.join(output_path, str(cam_idx + 1))), image)

            if extrinsic['H'] is None:
                continue
            undistorted_img = undistort_one(image, extrinsic['K'], extrinsic['distCoeff'], extrinsic['K'], extrinsic['validPixROI'])
            img_dst = homo_transform(undistorted_img, reference, homo_mat)
            cv2.imwrite('{}.png'.format(osp.join(output_path, str(cam_idx+1))), img_dst)


if __name__ == '__main__':
    main()

import numpy as np
import json
import cv2
np.set_printoptions(precision=4, suppress=True)

def save_json(content, pathname):
    """
    Save content in a json file
    :param content: a dictionary of np arrays
    :param pathname: pathname to save a json
    """
    content_json = {}
    for name, arr in content.items():
        if isinstance(arr, np.ndarray):
            shape = arr.shape
            content_json[name] = {
                'data': arr.reshape(-1).tolist(),
                'shape': shape
            }
        elif isinstance(arr, tuple) or isinstance(arr, list):
            content_json[name] = {
                'data': arr
            }
        elif name == 'reprojection_error':
            content_json[name] = arr
        else:
            raise NotImplementedError

    with open(pathname, 'w') as f:
        json.dump(content_json, f, indent=4)


with open('intrinsic-f.json', 'r') as file:
    data = json.load(file)['cameras']

for cam_ind in range(1, 7):
    cam = data['cam{}'.format(cam_ind)]
    output = {}
    output['camera_matrix'] = np.array(cam['K'])
    output['distortion_coefficients'] = np.array(cam['dist'])
    output['new_camera_matrix'] = np.array(cam['K'])
    output['region_of_interest'] = np.array([0, 0, 2856, 2848])
    save_json(output, '{}.json'.format(cam_ind))

print('done')
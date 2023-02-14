import bpy
import json
import numpy as np
from bpy import context

scene = context.scene
C = bpy.context


def generate_camera_matrices_from_json(path, id):
    with open(path, "r") as f:
        data = json.loads(f.read())

    data = data[id]
    intrinsic = np.asarray(data['K']).reshape(3, 3)

    extrinsic = np.zeros((4, 4))
    extrinsic[3, 3] = 1
    rotation = np.asarray(data['R']).reshape(3, 3)
    translation = np.asarray(data['T']).reshape(3, 1)

    rotation_camera = rotation.T
    camera_center = -rotation_camera @ translation
    extrinsic[:3, :3] = rotation_camera
    extrinsic[:3, -1:] = camera_center
    extrinsic[:3, :3] = -np.eye(3) @ extrinsic[:3, :3]

    image_size = data['imgSize']
    return intrinsic, extrinsic, image_size


def get_cameras(path):
    # add collection
    collection_name = 'cameras'
    if collection_name not in bpy.data.collections:
        col_cam = bpy.data.collections.new('cameras')
        bpy.context.scene.collection.children.link(col_cam)

    # add cameras
    for camera_id in range(20):
        intrinsic, extrinsic, img_size = generate_camera_matrices_from_json(path, camera_id)

        camera_data = bpy.data.cameras.new(name=str(camera_id))
        camera_data.display_size = 0.25
        camera_object = bpy.data.objects.new(str(camera_id), camera_data)

        # add to collection
        col_cam = C.scene.collection.children.get('cameras')
        col_cam.objects.link(camera_object)

        camera_object.matrix_world = extrinsic.T
        print(np.asarray(extrinsic))

        fx = intrinsic[0][0]
        fy = intrinsic[1][1]
        u0 = intrinsic[0][2]
        v0 = intrinsic[1][2]
        K = [[fx, 0, u0], [0, fy, v0], [0, 0, 1]]
        sensor_width = 36
        width = img_size[0]
        height = img_size[1]
        camera_object.data.lens = (K[0][0] + K[1][1]) / 2 * sensor_width / width
        camera_object.data.shift_x = (u0 - width / 2) / width
        camera_object.data.shift_y = (v0 - height / 2) / width

        scene.camera = camera_object
        # bpy.context.scene.render.filepath = f"/Users/daxuan/Desktop/{camera_id}.png"
        # bpy.ops.render.render(write_still=True)


def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3)  # cross of all zeros only occurs on identical directions


def rot2eul(R):
    beta = -np.arcsin(R[2, 0])
    alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
    gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
    return np.array((alpha, beta, gamma))


def get_frames(npy_path):
    data = np.load(npy_path, allow_pickle=True)
    key_points_all_frames = []
    for frame_id, d in enumerate(data):
        key_points = d["targets"]
        p8, p9, p16 = key_points[:, 7, :3], key_points[:, 8, :3], key_points[:, 15, :3]
        for i, p in enumerate(np.stack([p8, p9, p16], axis=1)):
            print("=-=-=-=")
            locations = p.mean(axis=0)
            headding_vec = np.cross((p[0] - p[1]), p[0] - p[2])
            rot_mat = rotation_matrix_from_vectors(np.array([1, 0, 0]), headding_vec)
            euler_angle = rot2eul(rot_mat)
            human = bpy.data.objects.get("human_mesh")

            human_new = human.copy()
            human_new.data = human.data.copy()
            human_new.animation_data_clear()
            C.collection.objects.link(human_new)
            rotation_homo = np.eye(4)
            rotation_homo[:3, :3] = rot_mat
            human_new.matrix_world = rotation_homo
            human_new.location = locations
            human_new.name = f"human_{i}"
        bpy.context.scene.render.filepath = f"/Users/daxuan/Desktop/frames/{frame_id:05d}.png"
        # bpy.ops.render.render(write_still=True)
        for i, p in enumerate(np.stack([p8, p9, p16], axis=1)):
            objs = bpy.data.objects
            objs.remove(objs[f"human_{i}"], do_unlink=True)


if __name__ == "__main__":
    path = 'D:/CameraCalibration/multical/sdk_parameters.json'
    #    path = 'D:/WTT/sdk_parameters.json'
    get_cameras(path)

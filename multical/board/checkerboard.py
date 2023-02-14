from multical.board.board import Board
from pprint import pformat
from cached_property import cached_property
import cv2
import numpy as np
from .common import *

from structs.struct import struct, choose, subset
from multical.optimization.parameters import Parameters

class CheckerBoard(Parameters, Board):

  def __init__(self, size=10, square_length=10, min_rows=3, min_points=20, adjusted_points=None):
    self.size = tuple(size)
    self.square_length = square_length 
    self.min_rows = min_rows
    self.min_points = min_points
    self.adjusted_points = choose(adjusted_points, self.points) 


  @cached_property
  def board(self):
    assert False, "Checkerboard generation is not currently not supported"

  def export(self):
    return struct(
      type='checkerboard',
      size = self.size,
      num_ids = self.size[0] * self.size[1],
      square_length = self.square_length
    )

  def __eq__(self, other):
    return self.export() == other.export()

  @property
  def points(self):
    points_ = []
    for i in range(self.size[1]):
      for j in range(self.size[0]):
        points_.append([i*self.square_length, j*self.square_length, 0])
    # assert False, "points not implemented"
    # import pdb; pdb.set_trace()
    return np.asarray(points_, dtype=np.float32)
    # return self.board.chessboardCorners
  
  @property
  def num_points(self):
    return 11*8 # 88

  @property 
  def ids(self):
    return np.arange(self.num_points)

  @cached_property
  def mesh(self):
    return grid_mesh(self.adjusted_points, self.size)
    # assert False, "mesh not implmented"
    # return grid_mesh(self.adjusted_points, self.size)

  @property
  def size_mm(self):
    square_length = int(self.square_length * 1000)
    return [dim * square_length for dim in self.size]


  def draw(self, pixels_mm=1, margin=20):
    assert False, "Darwing of board not implemented"


  def __str__(self):
      d = self.export()
      return "Checkerboard " + str(self.size)

  def __repr__(self):
      return self.__str__()      

  def detect(self, image):
    # cv2.imwrite("img.png", image)
    ret_val, corners = cv2.findChessboardCorners(image, self.size, None)
    if not ret_val: return empty_detection
    print("Refining corners")
    corners = cv2.cornerSubPix(image, corners, (10,10), (-2,2), criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001))
    return struct(corners = corners.squeeze(1), ids=np.arange(self.size[0]* self.size[1]))

  def has_min_detections(self, detections):
    return has_min_detections_grid(self.size, detections.ids, 
      min_points=self.min_points, min_rows=self.min_rows)

  def estimate_pose_points(self, camera, detections):
    return estimate_pose_points(self, camera, detections)


  @cached_property
  def params(self):
    return self.adjusted_points

  def with_params(self, params):
    return self.copy(adjusted_points = params)

  def copy(self, **k):
      d = self.__getstate__()
      d.update(k)
      return CheckerBoard(**d)

  def __getstate__(self):
    return subset(self.__dict__, ['size', 'square_length', 'min_rows', 'min_points', 'adjusted_points'])





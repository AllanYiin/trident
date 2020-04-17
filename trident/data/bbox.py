"""2D bounding box module."""
from copy import deepcopy
from math import cos, sin

import numpy as np


class bbox:
    """
    Class to reprsent a 2D rotatable bounding box.

    Args:
        pts: Sequence of length 4 representing (x, y, w, h) or (x1, y1, x2, y2) or ((x1, y1), (x2, y2)) or ((x1,y1),(x2,y2),(x3,y3),(x4,y4)).depending on ``mode``.
        mode (string): Indicator of box format (x, y, w, h) or (x1, y1, x2, y2) or (x1,y1,x2,y2,x3,y3,x4,y4). \
        The values are 0 for 'xywh' format and 1 for 'xyxy' format. See :py:mod:`~bbox.box_modes`.

    Raises:
        ValueError: If `x` is not of length 4.
        TypeError: If `x` is not of type {list, tuple, numpy.ndarray, Bbox}

    """
    def __init__(self, pts, mode='xyxy'):
        # Copy constructor makes the constructor idempotent
        _available_mode=['xywh','xyxy','corner']

        self._corner=np.zeros((4,2))
        self.probability=0
        self.class_idx=-1
        self.theta=0
        self._keypoints=None

        if isinstance(pts, np.ndarray) and pts.ndim >= 2:
            pts = pts.flatten()

        if mode=='xywh':
            if  isinstance(pts, (list, tuple,np.ndarray)):
                if len(pts) != 4 and len(pts) != 5and len(pts) != 6:
                    raise ValueError("Invalid input length. Input should have 4 elements.")
                else:
                    xc,yc,w,h=pts [:4]
                    self._corner[0]=(xc-(w/2.0),  yc-(h/2.0))
                    self._corner[1]=(xc+(w/2.0),yc-(h/2.0))
                    self._corner[2]=(xc+(w/2.0),  yc+(h/2.0))
                    self._corner[3]=(xc-(w/2.0),  yc+(h/2.0))
                    if len(pts) >=5:
                        self.probability=pts[4]
                    else:
                        self.probability=1
                    if len(pts) >=6:
                        self.classid=pts[5]

        elif mode=='xyxy':
            if  isinstance(pts, (list, tuple,np.ndarray)):
                if len(pts) != 2 and  len(pts) != 4 and len(pts) != 5and len(pts) != 6:
                    raise ValueError("Invalid input length. Input should have 4 elements.")
                else:
                    if len(pts) == 2 and isinstance(pts[0],(list,tuple)):
                        self._corner[0] = pts[0]
                        self._corner[1] = (pts[1][0], pts[0][1])
                        self._corner[2] = pts[1]
                        self._corner[3] = (pts[0][0], pts[1][1])
                    else:
                        x1,y1,x2,y2=pts [:4]
                        self._corner[0]=(x1,y1)
                        self._corner[1]=(x2,y1)
                        self._corner[2]=(x2,y2)
                        self._corner[3]=(x1,y2)
                        if len(pts) >=5:
                            self.probability=pts[4]
                        else:
                            self.probability=1
                        if len(pts) >=6:
                            self.classid=pts[5]
        elif mode == 'corner':
            if len(pts) >= 8  :
                self._corner[0] =(pts[0],pts[1])
                self._corner[1] =(pts[2],pts[3])
                self._corner[2] =(pts[4],pts[5])
                self._corner[3] =(pts[6],pts[7])


                if len(pts) >= 9 and 0<=pts[8]<=1:
                    self.probability = pts[8]
                else:
                    self.probability = 1
                if len(pts) >= 10:
                    self.classid = pts[9]
            elif len(pts) >= 5:
                x1,y1,x2,y2,theta=pts[:5]
                self.theta=theta
                xc=(x1+x2)/2.0
                yc=(y1+y2)/2.0
                for  i in range(len(self._corner)):
                    x,y=self._corner[i]
                    x2 = xc + (x - xc) * cos(theta) + (y - yc) * sin(theta)
                    y2 = yc - (x - xc) * sin(theta) + (y - yc) * cos(theta)
                    self._corner[i][0]=x2
                    self._corner[i][1]=y2
                if len(pts) >= 5:
                    self.probability = pts[4]
                else:
                    self.probability = 1
                if len(pts) >= 6:
                    self.classid = pts[5]



        if isinstance(pts, bbox):
            pts = pts.numpy(mode=mode)

        elif isinstance(pts, (list, tuple)):
            if len(pts) != 4and len(pts) != 5:
                raise ValueError("Invalid input length. Input should have 4 elements.")
            pts = np.asarray(pts)

        elif isinstance(pts, np.ndarray):
            if pts.ndim >= 2:
                pts = pts.flatten()
            if pts.size != 4 and pts.size != 5:
                raise ValueError(
                    "Invalid input length. Input should have 4 elements.")
        else:
            raise TypeError(
                "Expected input to constructor to be a 4 element " \
                    "list, tuple, numpy ndarray, or Bbox object.")
        self.xc = np.float((pts[2] + pts[0]) / 2)
        self.yc = np.float((pts[3] + pts[1]) / 2)
        if mode == 'xyxy':
            w = pts[2] - pts[0] + 1
            h = pts[3] - pts[1] + 1

        elif mode == 'xywh':
            w = pts[2]
            h = pts[3]
            self.xc = np.float(pts[0])
            self.yc = np.float(pts[1])
        else:
            raise ValueError('argument mode has invalid value')

        self.is_stop_check = False
        self.classid = np.float(pts[0])
        if len(pts)==5:
            self.classid = np.float(pts[4])
        self._x1 = np.float(pts[0])
        self._y1 = np.float(pts[1])
        self._w = np.float(w)
        self._h = np.float(h)

        # (x2, y2) will be used for indexing, hence we need to subtract 1
        self._x2 = self._x1 + self._w - 1
        self._y2 = self._y1 + self._h - 1


    def __eq__(self, x):
        if not isinstance(x, bbox):
            return False
        return self._x1 == x.x1 \
            and self._y1 == x.y1 \
            and self._x2 == x.x2 \
            and self._y2 == x.y2

    def angle_between(self,v1, v2):
        dot_pr = v1.dot(v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.rad2deg(np.arccos(dot_pr / norms))

    def to_xyxy(self):
        xs = self._corner[::2]
        ys = self._corner[1::2]
        return np.array([max(xs.min(),0),max(ys.min(),0),xs.max(),ys.max()]).round().astype(np.uint8)

    def to_x(self):
        xs = self._corner[::2]
        ys = self._corner[1::2]
        return np.array([xs.mean(),ys.mean(),xs.max()-xs.min(),ys.max()-ys.min()]).round().astype(np.uint8)

    @property
    def x1(self):
        """
        :py:class:`float`: Left x coordinate.
        """
        return max(self._x1,0)
    @x1.setter
    def x1(self, x):
        if self.is_stop_check ==False and x > self.x2:
            raise ValueError("Value is greater than x2={0}".format(self.x2))

        self._x1 = x
        self._w = self._x2 - self._x1 + 1
        self.xc = self._x1 + (self._w - 1) / 2

    @property
    def x2(self):
        """
        :py:class:`float`: Right x coordinate.
        """
        return self._x2

    @x2.setter
    def x2(self, x):
        if  self.is_stop_check ==False and x < self.x1:
            raise ValueError("Value is lesser than x1={0}".format(self.x1))

        self._x2 = x
        self._w = self._x2 - self._x1 + 1
        self.xc = self._x1 + (self._w - 1) / 2

    @property
    def y1(self):
        """
        :py:class:`float`: Top y coordinate.
        """
        return max(self._y1,0)

    @y1.setter
    def y1(self, y):
        if  self.is_stop_check ==False and  y > self.y2:
            raise ValueError("Value{0} is greater than y2={0}".format(self.y2))

        self._y1 = y
        self._h = self._y2 - self._y1 + 1
        self.yc = self._y1 + (self._h - 1) / 2

    @property
    def y2(self):
        """
        :py:class:`float`: Bottom y coordinate.
        """
        return self._y2

    @y2.setter
    def y2(self, y):
        if  self.is_stop_check ==False and y < self.y1:
            raise ValueError("Value is lesser than y1={0}".format(self.y1))

        self._y2 = y
        self._h = self._y2 - self._y1 + 1
        self.yc = self._y1 + (self._h - 1) / 2

    @property
    def width(self):
        """
        :py:class:`float`: Width of bounding box.
        """
        return self._w

    @width.setter
    def width(self, w):
        if w < 1:
            raise ValueError(
                "Invalid width value. Width cannot be non-positive.")
        self._w = w
        self._x2 = self._x1 + self._w - 1
        self.xc=self._x1 + (self._w - 1) / 2

    @property
    def w(self):
        """
        :py:class:`float`: Syntactic sugar for width.
        """
        return self._w

    @w.setter
    def w(self, w):
        self.width = w

    @property
    def height(self):
        """
        :py:class:`float`: Height of bounding box.
        """
        return self._h

    @height.setter
    def height(self, h):
        if h < 1:
            raise ValueError(
                "Invalid height value. Height cannot be non-positive.")
        self._h = h

        self._y2 = self._y1 + self._h - 1
        self.yc=self._y1 + (self._h-1)/2

    @property
    def h(self):
        """
        :py:class:`float`: Syntactic sugar for height.
        """
        return self._h

    @h.setter
    def h(self, h):
        self.height = h



    def center(self):
        """
        Return center coordinates of the bounding box.
        """
        return np.array([self._x1 + (self._w-1)/2, self._y1 + (self._h-1)/2])


    def aspect_ratio(self, ratio):
        """
        Return bounding box mapped to new aspect ratio denoted by ``ratio``.

        Args:
            ratio (:py:class:`float`): The new ratio should be given as \
                the result of `width / height`.
        """
        # we need ratio as height/width for the below formula to be correct
        ratio = 1.0 / ratio

        area = self.w * self.h
        area_ratio = area / ratio
        new_width = np.round(np.sqrt(area_ratio))
        new_height = np.round(ratio * new_width)
        new_bbox = bbox((self.x1, self.y1, new_width, new_height,self.classid),mode='xywh')
        return new_bbox

    # def get_'xywh'(self):
    #     return self.numpy('xywh')
    # def get_XXYY(self):
    #     return self.numpy('xywh')
    def corner(self):
        return np.asarray([[self._x1,self._y1,1],[self._x2,self._y1,1],[self._x1,self._y2,1],[self._x2,self._y2,1]])

    def tolist(self, mode='xyxy'):
        """
        Return bounding box as a `list` of 4 numbers.
        Format depends on ``mode`` flag (default is 'xywh').

        Args:
            mode (BoxMode2D): Mode in which to return the box. See :py:mod:`~bbox.box_modes`.
        """
        if mode=='xyxy':
            return [self.x1, self.y1, self.x2, self.y2,self.classid]

        return [self.x1, self.y1, self.w, self.h,self.classid]

    def copy(self):
        """
        Return a deep copy of this 2D bounding box.
        """
        return deepcopy(self)

    def numpy(self, mode='xyxy'):
        """
        Return bounding box as a numpy vector of length 4.
        Format depends on ``mode`` flag (default is 'xywh').

        Args:
            mode (BoxMode2D): Mode in which to return the box. See :py:mod:`~bbox.box_modes`.
        """
        return np.asarray(self.tolist(mode=mode), dtype=np.float)

    def __repr__(self):
        return "box([{x}, {y}, {x2}, {y2},{classid}])".format(x=self.x1, y=self.y1, x2=self.x2, y2=self.y2,classid=self.classid)

    def mul(self, s):
        """
        Multiply the box by a scalar. Used for scaling bounding boxes.

        Args:
            s (:py:class:`float` or `int`): Scalar value to scale the box by.
        """
        if not isinstance(s, (int, float)):
            raise ValueError(
                "bounding boxes can only be multiplied by scalar (int or float)")
        return bbox([self.x1 * s, self.y1 * s, self.x2 * s, self.y2 * s,self.classid], mode='xyxy')


    def resize(self,new_x1,new_y1,new_x2,new_y2):
        self.is_stop_check=True
        self._x1 = max(new_x1,0)
        self._y1 =max(new_y1,0)
        self._x2 = new_x2
        self._y2 = new_y2
        self._h = self._y2 - self._y1 + 1
        self._w = self._x2 - self._x1 + 1
        self.xc = self._x1 + (self._w - 1) / 2
        self.yc = self._y1 + (self._h - 1) / 2

        self.is_stop_check = False

    def __mul__(self, s):
        return self.mul(s)

    def __rmul__(self, s):
        return self.mul(s)
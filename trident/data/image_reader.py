from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import Queue
except ImportError:
    import queue as Queue
import threading
import time

from ..misc.ipython_utils import is_in_ipython, is_in_colab

if is_in_ipython():
    from IPython import display

if not is_in_colab:
    import matplotlib
    matplotlib.use('Qt5Agg' if not is_in_ipython() and not is_in_colab() else 'NbAgg' )
else:
    import matplotlib
import matplotlib.pyplot as plt
import itertools
from .image_common import list_pictures



class ImageThread(threading.Thread):
    """Image Thread"""
    def __init__(self, queue, out_queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.out_queue = out_queue

    def run(self):
        while True:
            # Grabs image path from queue
            image_path_group = self.queue.get()
            # Grab image
            image_group = [plt.imread(i) for i in image_path_group]
            # Place image in out queue
            self.out_queue.put(image_group)
            # Signals to queue job is done
            self.queue.task_done()

class ImageReader(object):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self,images=None):
        self.image_paths =None
        if images is not None :
            if hasattr(images,' __iter__') and all(isinstance(images, str) for img in images):
                self.image_paths=images
            else:
                raise TypeError('pins must be a list of one or more strings.')

        self.workers=2
        self.itr = 0
        self.statistics=[]
        self.buffer_size = 5
        self._minibatch_size = 32
        self.input_qsize = 50
        self.min_input_qsize = 10
        self.n_minibatches_to_run = float('inf')
        self.queue = Queue.Queue()
        self.out_queue = Queue.Queue(maxsize=self.buffer_size)

        self.prepare_queue()

    def prepare_queue(self):
        if self.image_paths is not  None and len(self.image_paths)>0:
            self.grouped_image_paths = zip(*[iter(self.image_paths[:-(len(self.image_paths) % self._minibatch_size)])] * self._minibatch_size)
            self.grouped_image_paths = itertools.cycle(self.grouped_image_paths)

            self.threadPool=[]
            for i in range(self.workers):
                t = ImageThread(self.queue, self.out_queue)
                t.setDaemon(True)
                t.start()
                self.threadPool.append(t)
            for image_path_group in range(self.input_qsize):
                image_path_group = self.grouped_image_paths.__next__()
                self.queue.put(image_path_group)

    @property
    def minibatch_size(self):
        return self._minibatch_size

    @minibatch_size.setter
    def minibatch_size(self, minibatch_size):
        if (isinstance(minibatch_size, str)):
            self._minibatch_size = int(minibatch_size)
        elif (isinstance(minibatch_size, int)):
            self._minibatch_size = minibatch_size
        self.grouped_image_paths = zip(*[iter(self.image_paths[:-(len(self.image_paths) % self._minibatch_size)])] * self._minibatch_size)
        self.grouped_image_paths = itertools.cycle(self.grouped_image_paths)

    def get_all_images(self,base_folder):
        self.image_paths=list_pictures(base_folder)
        self.prepare_queue()

    def __iter__(self):
        if self.itr<=self.n_minibatches_to_run:
            start = time.time()
            image_group = self.out_queue.get()
            stop = time.time()
            self.statistics.append(stop - start)
            self.itr += 1
            if self.queue.qsize() <= self.min_input_qsize:
                for image_path_group in range(self.input_qsize):
                    image_path_group = self.grouped_image_paths.__next__()
                    self.queue.put(image_path_group)
            yield image_group




    def __len__(self):
        return len(self.image_paths) -(len(self.image_paths) % self._minibatch_size)






try:
    import Queue
except ImportError:
    import queue as Queue
import threading
import time
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import itertools
import random

# jpg test
base_path = "/data/lisa/data/COCO/images/train2014/"
image_paths = glob.glob(os.path.join(base_path, "*.jpg"))

# Test random order
random.shuffle(image_paths)

buffer_size = 5
minibatch_size = 10
input_qsize = 50
min_input_qsize = 10
n_minibatches_to_run = 100
if len(image_paths) % minibatch_size != 0:
    print("WARNING: Sample size not an even multiple of minibatch size")
    print("Truncating...")
    image_paths = image_paths[:-(len(image_paths) % minibatch_size)]
assert len(image_paths) % minibatch_size == 0

grouped_image_paths = zip(*[iter(image_paths)] * minibatch_size)
# Infinite...
grouped_image_paths = itertools.cycle(grouped_image_paths)

queue = Queue.Queue()
out_queue = Queue.Queue(maxsize=buffer_size)


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
            # print('reading image', image_path)
            image_group = [plt.imread(i) for i in image_path_group]
            # Place image in out queue
            self.out_queue.put(image_group)
            # Signals to queue job is done
            self.queue.task_done()


def threaded_main():
    for i in range(1):
        t = ImageThread(queue, out_queue)
        t.setDaemon(True)
        t.start()

    # Populate queue with some paths to image data
    for image_path_group in range(input_qsize):
        image_path_group = grouped_image_paths.next()
        queue.put(image_path_group)

    start = time.time()
    itr = 1
    while True:
        image_group = out_queue.get()
        # time.sleep approximates running some model
        time.sleep(1)
        stop = time.time()
        tot = stop - start
        print("Threaded time: %s" % (tot))
        print("Minibatch %s" % str(itr))
        print("Time ratio (s per minibatch): %s" % (tot / float(itr)))
        itr += 1
        if queue.qsize() <= min_input_qsize:
            for image_path_group in range(input_qsize):
                image_path_group = grouped_image_paths.next()
                queue.put(image_path_group)
        # test
        if itr >= n_minibatches_to_run:
            break


def unthreaded_main():
    start = time.time()
    itr = 1
    for image_path_group in grouped_image_paths:
        image_group = [plt.imread(i) for i in image_path_group]
        # time.sleep approximates running some model
        time.sleep(1)
        stop = time.time()
        tot = stop - start
        print("Unthreaded time: %s" % (tot))
        print("Minibatch %s" % str(itr))
        print("Time ratio (s per minibatch): %s" % (tot / float(itr)))
        itr += 1

        # test
        if itr >= n_minibatches_to_run:
            break

# Randomize order to avoid caching
if int(time.time()) % 2 == 0:
    print("Running unthreaded, then threaded")
    unthreaded_main()
    threaded_main()
else:
    print("Running threaded, then unthreaded")
    threaded_main()
    unthreaded_main()
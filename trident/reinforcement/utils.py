from collections import namedtuple
from trident.backend.pytorch_ops import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):
    """ A buffer to hold previously-generated states. """
    def __init__(self, capacity):
        """

        Args:
            capacity ()int: Max number of transitions to store in the buffer. When the buffer overflows the old memories are dropped.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0



    def push(self, *args):
        """Saves a transition."""
        args = [to_numpy(arg) for arg in args]
        if len(self.memory) < self.capacity:

            self.memory.append( Transition(*args))
        else:
            self.memory[self.position]=Transition(*args)

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Query the memory to construct batch"""
        transitions=random_choice(self.memory, batch_size) #list of named tuple [(x1,y1),(x2,y2),(x3,y3)]
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        items=list(zip(*transitions))
        items=[item for item in items ]
        return  Transition(*items)

    def __len__(self):
        return len(self.memory)
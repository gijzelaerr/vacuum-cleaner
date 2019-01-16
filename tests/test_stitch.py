from unittest import TestCase
from vacuum.stitch import init, clean
import numpy as np

CHECKPOINT = '/scratch/gijs/vacuum-cleaner/train/landman_likelihood'


class TestStitch(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = init(CHECKPOINT)

    def test_clean_uneven(self):
        dirty = np.eye(513)
        psf = np.eye(513)
        clean(self.model, dirty, psf)

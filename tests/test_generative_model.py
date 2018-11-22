from os import path
from vacuum.datasets import generative_model

here = path.dirname(__file__)
test_dir = path.join(here, 'data')


def test_generative_model():
    generative_model(path.join(test_dir, '*-bigpsf-psf.fits'))

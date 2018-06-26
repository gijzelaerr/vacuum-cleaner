from setuptools import setup, find_packages

__version__ = "0.1.3"


install_requires = [
    'tensorflow',
    'astropy',
]

extras_require = {
    'gpu': ['tensorflow-gpu'],
}

data_files = [
    ('share/vacuum/model', [
        'share/vacuum/model/model-30000.meta',
        'share/vacuum/model/model-30000.index',
        'share/vacuum/model/checkpoint',

    ])
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='vacuum-cleaner',
    version=__version__,
    packages=find_packages(),
    data_files=data_files,
    install_requires=install_requires,
    extras_require=extras_require,
    author="Gijs Molenaar",
    author_email="gijs@pythonic.nl",
    description="Deep Vacuum Cleaner",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords="radio astronomy deep learning cleaning science",
    url="https://github.com/gijzelaerr/vacuum-cleaner",
    classifiers=[
                  "License :: OSI Approved :: MIT License",
                  "Development Status :: 3 - Alpha",
                  "Programming Language :: Python",
                  "Programming Language :: Python :: 3.6",
                  "Programming Language :: Python :: 3.7",
                  "Intended Audience :: Science/Research",
                  "Topic :: Scientific/Engineering :: Artificial Intelligence",
                  "Topic :: Scientific/Engineering :: Astronomy",
              ],
    entry_points={
      'console_scripts': [
        'vacuum-cleaner = vacuum.cleaner:main',
        ]
    }
)

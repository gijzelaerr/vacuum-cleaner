from setuptools import setup, find_packages

__version__ = "0.1.1"


install_requires = [
    'tensorflow',
    'astropy',
]

extra_require = {
    'gpu': 'tensorflow-gpu'
}

data_files = [
    ('share/vacuum/model', [
        # 'share/vacuum/model/model-30000.data-00000-of-00001',
        'share/vacuum/model/model-30000.meta',
        'share/vacuum/model/model-30000.index',
        'share/vacuum/model/checkpoint',

    ])
]

setup(
    name='vacuum-cleaner',
    version=__version__,
    packages=find_packages(),
    data_files=data_files,
    install_requires=install_requires,
    extra_require=extra_require,
    author="Gijs Molenaar",
    author_email="gijs@pythonic.nl",
    description="Deep Vacuum Cleaner",
    license="MIT",
    # setup_requires=['pytest-runner'],
    # tests_require=['pytest', 'mock'],
    # test_suite="tests",
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
        # 'vacuum = vacuum.cleaner:main',
        'vacuum-cleaner = vacuum.cleaner:main',
        ]
    }
)

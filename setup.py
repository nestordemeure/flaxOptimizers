# https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56
from distutils.core import setup

setup(
    name = 'flaxOptimizers',
    packages = ['flaxOptimizers'],
    version = '1.0',
    license = 'apache-2.0',
    description = 'My optimizer implementations for Flax.',
    author = 'NestorDemeure',
    # author_email = 'your.email@domain.com',
    url = 'https://github.com/nestordemeure/hyperknee',
    # download_url = 'https://github.com/nestordemeure/AdaHessianJax/archive/v?.?.tar.gz',
    keywords = ['deep-learning', 'optimizer', 'flax'],
    install_requires=['jax', 'flax'],
    classifiers=[ # https://pypi.org/classifiers/
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
    ],
)

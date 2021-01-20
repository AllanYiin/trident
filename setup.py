from setuptools import setup, find_packages, Extension



from pkg_resources import get_distribution, DistributionNotFound
import subprocess
import distutils.command.clean
import distutils.spawn
import glob
import shutil
import os

# with open("README.md", "r",encoding='utf-8-sig') as fh:
#     long_description = fh.read()



NAME = "tridentx"
DIR = '.'
EXCLUDE_FROM_PACKAGES = ["tests", "examples","internal_tool","datav3"]
PACKAGES = find_packages(exclude=EXCLUDE_FROM_PACKAGES)


setup(name=NAME,
      version='0.7.1',
      description='Make pytorch and tensorflow two become one.',
      # long_description=long_description,
      # long_description_content_type="text/markdown",
      long_description=open("README.md", encoding="utf8").read(),
      long_description_content_type="text/markdown",
      author= 'Allan Yiin',
      author_email= 'allanyiin.ai@gamil.com',
      download_url= 'https://test.pypi.org/project/tridentx',
      license='MIT',
      install_requires=['numpy>=1.18',
                        'scikit-image >= 0.15',
                        'pillow >= 4.1.1',
                        'scipy>=1.2',
                        'six>=1.13.0',
                        'matplotlib>=3.0.2',
                        'tensorboard>=1.15',
                        'opencv-python',
                        'setuptools',
                        'tqdm',
                        'pyyaml',
                        'h5py',
                        'dill',
                        'requests'],
      extras_require={
          'visualize': ['pydot>=1.2.4',],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'flaky',
                    'pytest-cov',
                    'requests',
                    'markdown'],
      },
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      python_requires='>=3',

      packages=PACKAGES,
      include_package_data=True,
      scripts=[],

      )


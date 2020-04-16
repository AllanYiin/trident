from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='tridentx',
      version='0.0.4',
      description='Multiverse for Deep Learning Developers',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author= 'Allan Yiin',
      author_email= 'allan@datadecision.ai',
      url= '....',
      download_url= '....',
      license='MIT',
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'six>=1.9.0',
                        'tqdm',
                        'pyyaml',
                        'h5py',
                        'requests', 'django-layers-hr'],
      extras_require={
          'visualize': ['pydot>=1.2.4'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'flaky',
                    'pytest-cov',
                    'pandas',
                    'requests',
                    'markdown'],
      },
      classifiers=[
          'Development Status :: 3 - Alpha',
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
      packages=find_packages())


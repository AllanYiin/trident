import os.path
import pathlib
import os
import pkg_resources
from setuptools import setup, find_packages

# with open("README.md", "r",encoding='utf-8-sig') as fh:
#     long_description = fh.read()



NAME = "tridentx"
DIR = '.'

PACKAGES = find_packages(exclude= ["tests","tests.*","sphinx_docs","sphinx_docs.*", "examples","examples.*","internal_tool","internal_tool.*"])
print(PACKAGES)




with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]




setup(name=NAME,
      version='0.7.9',
      description='Make pytorch and tensorflow two become one.',
      # long_description=long_description,
      # long_description_content_type="text/markdown",
      long_description=open("README.md", encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      author='Allan Yiin',
      author_email='allanyiin.ai@gmail.com',
      download_url='https://test.pypi.org/project/tridentx',
      license='MIT',
      install_requires=install_requires,
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
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      python_requires='>=3.5',
      keywords=['deep learning', 'machine learning', 'pytorch', 'tensorflow', 'AI'],
      packages= find_packages(exclude= ["tests","tests.*","sphinx_docs","sphinx_docs.*", "examples","examples.*","internal_tool","internal_tool.*"]),
      package_data={
          'trident': ['data/*.txt','models/*.txt'],
       },
      include_package_data=True,
      scripts=[],

      )


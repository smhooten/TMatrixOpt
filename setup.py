from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='TMatrixOpt',
   version='0.1',
   description='T-Matrix based adjoint-method optimizer',
   license="BSD-3-Clause",
   long_description=long_description,
   author='Sean Hooten',
   author_email='shooten@eecs.berkeley.edu',
   url="https://github.com/smhooten/TMatrixOpt",
   packages=['TMatrixOpt'],
   #install_requires=['numpy', 'scipy'],
   zip_safe=False,
)


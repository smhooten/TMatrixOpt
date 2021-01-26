from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='TMatrixOpt',
   version='1.0',
   description='T-Matrix based adjoint-method optimizer',
   license="GNU GPL 3.0",
   long_description=long_description,
   author='Sean Hooten, Zunaid Omair',
   author_email='shooten@eecs.berkeley.edu',
   url="https://github.com/smhooten/TMatrixOpt",
   packages=['TMatrixOpt'],
   package_data={'TMatrixOpt':['*.so']},
   zip_safe=False,
)


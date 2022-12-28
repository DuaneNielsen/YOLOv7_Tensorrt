#!/usr/bin/env python3
  
from setuptools import setup, find_packages


setup(name='YOLOv7_Tensorrt',
      version='1.0',
      description='converts yolov7 network to Tensor RT',
      author='Duane Nielsen',
      author_email='duane.nielsen.rocks@gmail.com',
      packages=find_packages('.'),
      # entry_points={'console_scripts': ['conefinder=conefinder.scripts.main:main']},
      install_requires=['tqdm','matplotlib', 'seaborn', 'scipy', 'onnx'],
     )

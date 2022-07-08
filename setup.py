from setuptools import setup, find_packages

setup(
    name='harmonia_classifier',
    install_requires=[
        'torch', 
        'torchvision', 
        'cudatoolset',
        'numpy',
        'matplotlib'
        ],
    packages=find_packages()
)
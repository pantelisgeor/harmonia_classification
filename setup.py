from setuptools import setup, find_packages

setup(
    name='harmonia_classifier',
    install_requires=[
        'torch', 
        'torchvision', 
        'cudatoolset'
        ],
    packages=find_packages()
)
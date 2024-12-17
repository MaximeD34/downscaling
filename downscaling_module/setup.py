from setuptools import setup, find_packages

setup(
    name="downscaling_module",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
    ],
)

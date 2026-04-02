from setuptools import setup, find_packages

setup(
    name='reskayl',
    version='0.1.0',
    description='ReSkayl - SRGAN Image Upscaler',
    author='Jaden',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'reskayl=src.cli:main',
        ],
    },
    install_requires=[
        'torch',
        'torchvision',
        'opencv-python',
        'numpy',
        'tqdm'
    ],
    python_requires='>=3.8',
)

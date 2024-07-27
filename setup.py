# setup.py
from setuptools import setup, find_packages

setup(
    name='your_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.0',
        # other dependencies
    ],
    author='Your Name',
    description='A block-based framework for TensorFlow',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your_package',
)

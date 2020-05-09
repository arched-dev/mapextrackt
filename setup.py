from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='mapextrackt',
    version='0.4',
    packages=['MapExtrackt'],
    url='https://github.com/lewis-morris/mapextrackt',
    license='MIT',
    author='Lewis Morris',
    author_email='lewis.morris@gmail.com',
    description='Pytorch Feature Map Extractor',
    long_description=long_description,
    long_description_content_type='text/markdown'
)

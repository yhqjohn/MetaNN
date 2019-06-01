from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

readme = open('README.rst').read()

setup(
    name='MetaNN',
    version='0.1.1',
    description='MetaNN provides extensions of PyTorch nn.Module for meta learning',
    license='MIT',
    author='Zhi Zhang',
    author_email='yhqjohn@gmail.com',
    keywords=['pytorch', 'meta learning'],
    url='https://github.com/yhqjohn/MetaModule',
    packages=find_packages(exclude=['tests']),
    long_description=readme,
    setup_requires=requirements
)

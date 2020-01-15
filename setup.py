from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

readme = open('README.rst').read()

setup(
    name='MetaNN',
    version='0.2.4',
    description='MetaNN provides extensions of PyTorch nn.Module for meta learning',
    author='Hanqiao Yu',
    author_email='yhqjohn@gmail.com',
    keywords=['pytorch', 'meta learning'],
    url='https://github.com/yhqjohn/MetaModule',
    packages=find_packages(exclude=['tests']),
    long_description=readme,
    setup_requires=requirements,
    classifiers=[
        'License :: OSI Approved :: MIT License',
    ],
)

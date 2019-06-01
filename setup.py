from setuptools import setup, find_packages

requirements = [
    'numpy',
    'torch',
]

readme = open('README.rst').read()

setup(
    name='MetaModule',
    version='0.1.0',
    description='MetaModule provides extensions of PyTorch Module for meta learning',
    license='MIT',
    author='Zhi Zhang',
    author_email='Hanqiao Yu',
    keywords=['pytorch', 'meta learning'],
    url='',
    packages=find_packages(exclude=['tests']),
    long_description=readme,
    setup_requires=requirements
)
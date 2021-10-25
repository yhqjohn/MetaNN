import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='MetaNN',
    version='0.2.10',
    description='MetaNN provides extensions of PyTorch nn.Module for meta learning',
    author='Hanqiao Yu',
    author_email='yhqjohn@gmail.com',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/yhqjohn/MetaModule",
    project_urls={
        "Bug Tracker": "https://github.com/yhqjohn/MetaModule/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=['tests']),
    python_requires=">=3.6",
    install_requires=[
        'torch>=0.4.1',
    ]
)

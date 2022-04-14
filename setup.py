from setuptools import setup

setup(
    name="svcd",
    version="0.1.1",    
    description="Spatially Varying Color Distributions for Interactive Multilabel Segmentation",
    author="Markus Plack, Hannah Dröge",
    license="MIT License",
    packages=["svcd"],
    install_requires=[
        "numpy", 
        "Pillow",
        "scipy",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",      
        "Programming Language :: Python :: 3",
    ],
)

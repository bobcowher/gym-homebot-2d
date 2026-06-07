from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gym-homebot-2d",
    version="0.1.0",
    author="Robert Cowher",
    author_email="bobcowher@gmail.com",
    description="A Gymnasium environment for a top-down 2D home robot task simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "scripts.*", "docs", "docs.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "gymnasium>=0.26.0",
        "pygame>=2.1.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "rl": ["torch>=1.9.0", "tensorboard>=2.8.0"],
        "dev": ["pytest>=7.0.0"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "homebot-play=play:main",
        ],
    },
)

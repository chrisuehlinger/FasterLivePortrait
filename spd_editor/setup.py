#!/usr/bin/env python
"""
Setup script for SPD Editor.
"""
# Set a fixed version instead of trying to read it from the file
# This avoids circular import issues during installation

VERSION = "0.1.0"

from setuptools import setup, find_packages

setup(
    name="spd_editor",
    version=VERSION,
    description="Source Portrait Descriptor Editor for FasterLivePortrait",
    author="FasterLivePortrait Team",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.7",
    install_requires=[
        # Core dependencies will be added here
    ],
    entry_points={
        "console_scripts": [
            "spd-editor=spd_editor.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
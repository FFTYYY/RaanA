from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import platform
from pkg_resources import parse_requirements
import pybind11

pybind11_include = pybind11.get_include()

ext_modules = [
    Extension(
        "raana.dp.dp",
        ["raana/dp/dp.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True)
        ], 
        language="c++"
    ),
    Extension(
        "raana.rabitq.rabitq",
        ["raana/rabitq/rabitq.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True)
        ], 
        language="c++"
    ),
]

setup(
    ext_modules=ext_modules,
)
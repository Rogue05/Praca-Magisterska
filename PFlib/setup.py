from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "PFlib",
        # sorted(glob("*.cpp")),  # Sort source files for reproducibility
        ["particle_filter.cpp",],
    ),
]

setup(
    name="PFlib",
    # extra_compile_args=['-O0','-Wall','-g'],
    ext_modules=ext_modules,
)
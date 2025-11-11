from setuptools import setup
import re

_version_re = re.compile(r"(?<=^__version__ = (\"|'))(.+)(?=\"|')")

def get_version(rel_path: str) -> str:
    """
    Searches for the ``__version__ = `` line in a source code file.

    https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
    """
    with open(rel_path, 'r') as f:
        matches = map(_version_re.search, f)
        filtered = filter(lambda m: m is not None, matches)
        version = next(filtered, None)
        if version is None:
            raise RuntimeError(f'Could not find __version__ in {rel_path}')
        return version.group(0)


setup(
    name="altastata-chris-demo",
    version=get_version("app.py"),
    description="AltaStata reference training plugin for the ChRIS platform",
    author="AltaStata",
    author_email="support@altastata.com",
    url="https://github.com/SergeVil/altastata-chris-demo",
    py_modules=["app"],
    install_requires=[
        "chris_plugin>=0.4.0",
        "altastata>=0.1.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "py4j>=0.10.9",
    ],
    license="MIT",
    entry_points={
        "console_scripts": [
            "altastata-chris-demo = app:main",
        ]
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    extras_require={
        "dev": [
            "pytest>=7.1",
        ]
    },
)

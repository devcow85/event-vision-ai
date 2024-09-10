from setuptools import setup, find_packages

setup(
    name="event-vision-ai",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "tonic",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "event-vision-ai=eva.main:main",
        ],
    },
)
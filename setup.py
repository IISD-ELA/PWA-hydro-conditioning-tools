from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='pwa-tools',
    version='0.1',
    description='Python module for hydro conditioning Prairie watersheds.',
    author='IISD-ELA',
    packages=find_packages(),
    install_requires=requirements,
)
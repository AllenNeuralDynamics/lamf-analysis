from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='lamf_analysis',
    version='0.1.5',
    url='https://github.com/AllenNeuralDynamics/ophys-mfish-dev',
    author='Matthew J. Davis',
    author_email='mattjdavis@gmail.com',
    description='TODO',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=requirements,
)

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ophys-mfish-dev',
    version='0.1.3',
    url='https://github.com/AllenNeuralDynamics/ophys-mfish-dev',
    author='Matthew J. Davis',
    author_email='mattjdavis@gmail.com',
    description='TODO',
    package_dir={'': 'ophys-mfish-dev'},
    packages=find_packages(where='ophys-mfish-dev'),
    install_requires=requirements,
)
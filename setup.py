from setuptools import setup, find_packages

setup(
    name='gymtonic',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'gymnasium',
        'pybullet'
    ],
    author='Inaki Vazquez',
    author_email='ivazquez@deusto.es',
    description='A set of Pybullet-based Gymnasium compatible environments',
    url='https://github.com/inakivazquez/gymtonic',
)
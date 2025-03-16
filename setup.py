from setuptools import setup, find_packages

setup(
    name='gymtonic',
    version='0.1.4',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.4',
        'gymnasium==0.29.1',
        'pybullet'
    ],
    package_data={
    # Specify the meshes folder
    "gymtonic.envs": ["meshes/*"],
    },
    author='Inaki Vazquez',
    author_email='ivazquez@deusto.es',
    description='A set of Pybullet-based Gymnasium compatible environments',
    url='https://github.com/inakivazquez/gymtonic',
)
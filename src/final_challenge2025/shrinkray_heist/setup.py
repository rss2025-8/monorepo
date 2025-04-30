import glob
import os
from setuptools import find_packages
from setuptools import setup

package_name = 'shrinkray_heist'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/shrinkray_heist/launch/sim', glob.glob(os.path.join('launch', 'sim', '*launch.*'))),
        ('share/shrinkray_heist/launch/real', glob.glob(os.path.join('launch', 'real', '*launch.*'))),
        ('share/shrinkray_heist/launch/debug', glob.glob(os.path.join('launch', 'debug', '*launch.*'))),
        (os.path.join('share', package_name, 'config', 'sim'), glob.glob('config/sim/*.yaml')),
        (os.path.join('share', package_name, 'config', 'real'), glob.glob('config/real/*.yaml')),
        (os.path.join('share', package_name, 'config', 'debug'), glob.glob('config/debug/*.yaml')),
        ('share/shrinkray_heist/example_trajectories', glob.glob(os.path.join('example_trajectories', '*.traj')))],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sebastian',
    maintainer_email='sebastianag2002@gmail.com',
    description='Path Planning ROS2 Package',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_builder = shrinkray_heist.trajectory_builder:main',
            'trajectory_loader = shrinkray_heist.trajectory_loader:main',
            'trajectory_planner = shrinkray_heist.trajectory_planner:main',
            'trajectory_follower = shrinkray_heist.trajectory_follower:main',
            'realistic_ackermann = shrinkray_heist.realistic_ackermann:main',
        ],
    },
)
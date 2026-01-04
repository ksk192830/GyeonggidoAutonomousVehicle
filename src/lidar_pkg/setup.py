from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'lidar_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sg',
    maintainer_email='sg@todo.todo',
    description='Lidar processing package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    
    entry_points={
        'console_scripts': [
            'scan = lidar_pkg.scan_publisher:main',
            'scan_viz = lidar_pkg.scan_viz:main',
            'scan_cluster = lidar_pkg.scan_cluster:main',
        ],
    },
)

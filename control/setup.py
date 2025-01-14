from setuptools import find_packages, setup

package_name = 'control'

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
    maintainer='chris',
    maintainer_email='23589086@sun.ac.za',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit = control.pure_pursuit:main',
            'pure_pursuit_c = control.pure_pursuit_c:main',
            'stanley = control.stanley:main',
            'follow_the_gap = control.follow_the_gap:main',
            'follow_the_gap_rl = control.follow_the_gap_rl:main',
            'mpc = control.mpc:main',
            'mpc2 = control.mpc_template:main',
            'mpc3 = control.mpc_tmp:main',
        ],
    },
)

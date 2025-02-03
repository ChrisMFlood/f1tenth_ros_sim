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
            'stanley = control.stanley:main',
            'stanley2 = control.stanley_2:main',
            'pure_pursuit = control.pure_pursuit:main',
            # 'pure_pursuit_c = control.pure_pursuit_c:main',
            'follow_the_gap = control.follow_the_gap:main',
            # 'follow_the_gap_rl = control.follow_the_gap_rl:main',
            # 'mpc_dynamic = control.mpc_dynamic:main',
            'mpc_kinematic = control.mpc_kinematic:main',
            # 'mpc = control.mpc:main',
            # 'mpc2 = control.mpc_template:main',
            # 'mpc3 = control.mpc_tmp:main',
            'wall_follow = control.wall_follow_node:main',
            'real_pure_pursuit = control.real_pure_pursuit:main',
            'real_stanley = control.real_stanley:main',
            'real_mpc = control.real_mpc:main',
        ],
    },
)

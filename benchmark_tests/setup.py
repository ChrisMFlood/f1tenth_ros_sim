from setuptools import find_packages, setup

package_name = 'benchmark_tests'

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
            "localisation_test_accuracy = benchmark_tests.localisation_test_accuracy:main",
            "scan_noise = benchmark_tests.scan_noise:main",
        ],
    },
)

from glob import glob

from setuptools import find_packages, setup

package_name = "radcounter_bringup"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RadCounterSim contributors",
    maintainer_email="maintainers@example.invalid",
    description="ROS 2 Jazzy bringup for RadCounterSim",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "radcounter_bridge = radcounter_bringup.bridge_node:main",
        ]
    },
)

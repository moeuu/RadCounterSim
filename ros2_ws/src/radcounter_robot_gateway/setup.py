from glob import glob

from setuptools import find_packages, setup

package_name = "radcounter_robot_gateway"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Eiji Morita",
    maintainer_email="morita@example.com",
    description="Nav2/MoveIt/Isaac countermeasure gateway",
    license="Apache-2.0",
    entry_points={
        "console_scripts": ["robot_gateway = radcounter_robot_gateway.gateway_node:main"]
    },
)

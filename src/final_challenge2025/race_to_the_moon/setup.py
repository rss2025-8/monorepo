import glob
import os

from setuptools import setup

package_name = "race_to_the_moon"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        # ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "lib/" + package_name + "/computer_vision",
            glob.glob(os.path.join("race_to_the_moon/computer_vision", "*.py")),
        ),
        ("share/race_to_the_moon/launch", glob.glob(os.path.join("launch", "*launch.xml"))),
        ("share/race_to_the_moon/launch", glob.glob(os.path.join("launch", "*launch.py"))),
        ("share/race_to_the_moon/config", glob.glob(os.path.join("config", "*.yaml"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Final Challenge Part B ROS2 package",
    license="Apache License, Version 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pure_pursuit = race_to_the_moon.pure_pursuit:main",
            "hough_line_lane_detector = race_to_the_moon.hough_line_lane_detector:main",
            "lane_homography_transformer = race_to_the_moon.lane_homography_transformer:main",
        ],
    },
)

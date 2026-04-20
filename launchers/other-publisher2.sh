#!/bin/bash
source /environment.sh
dt-launchfile-init
rosrun my_package other_publisher_node2.py
dt-launchfile-join
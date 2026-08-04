#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repository_root/scripts/host_env.sh"

prefix="${RADCOUNTER_ROS_PREFIX:-/home/moeu/.local/ros2/jazzy}"
work="${TMPDIR:-/tmp}/radcounter-ros-motion-debs"
packages=(
  ros-jazzy-control-msgs
  ros-jazzy-geographic-msgs
  ros-jazzy-moveit-msgs
  ros-jazzy-nav2-msgs
  ros-jazzy-object-recognition-msgs
  ros-jazzy-octomap-msgs
  ros-jazzy-shape-msgs
  ros-jazzy-trajectory-msgs
)

rm -rf "$work"
mkdir -p "$work/lists/partial" "$work/cache/archives/partial" "$work/debs" "$work/root"
curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o "$work/ros.key"
printf 'deb [arch=amd64 signed-by=%s] http://packages.ros.org/ros2/ubuntu noble main\n' "$work/ros.key" > "$work/ros2.list"
apt_options=(
  -o "Dir::Etc::sourcelist=$work/ros2.list"
  -o "Dir::Etc::sourceparts=-"
  -o "Dir::State::lists=$work/lists"
  -o "Dir::Cache=$work/cache"
  -o "APT::Get::List-Cleanup=0"
  -o Acquire::ForceIPv4=true
)
apt-get "${apt_options[@]}" update -qq
pushd "$work/debs" >/dev/null
for package in "${packages[@]}"; do
  apt-get "${apt_options[@]}" download "$package" >/dev/null
done
for archive in ./*.deb; do
  dpkg-deb -x "$archive" "$work/root"
done
popd >/dev/null
cp -a "$work/root/opt/ros/jazzy/." "$prefix/"
printf 'Installed ROS 2 Jazzy Nav2/MoveIt interface packages into %s\n' "$prefix"

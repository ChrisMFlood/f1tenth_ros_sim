# Download
## Quick Start
```bash
git clone
colcon buld
```
## F1tenth Bridge
```bash
cd $HOME
git clone https://github.com/f1tenth/f1tenth_gym
cd f1tenth_gym && pip3 install -e .

cd $HOME && mkdir -p sim_ws/src

cd $HOME/sim_ws/src
git clone https://github.com/f1tenth/f1tenth_gym_ros
```
- Update correct parameter for path to map file: Go to `sim.yaml` [https://github.com/f1tenth/f1tenth_gym_ros/blob/main/config/sim.yaml](https://github.com/f1tenth/f1tenth_gym_ros/blob/main/config/sim.yaml) in your cloned repo, change the `map_path` parameter to point to the correct location. It should be `'<your_home_dir>/sim_ws/src/f1tenth_gym_ros/maps/levine'`
```bash
source /opt/ros/humble/setup.bash
cd ..
rosdep install -i --from-path src --rosdistro humble -y
colcon build
source ~/sim_ws/install/setup.bash
echo "source ~/sim_ws/install/setup.bash" >> ~/.bashrc
```
## Pipeline 
### Perception
#### F1tenth Particle Filter
```bash
cd $HOME/sim_ws/src
git clone https://github.com/f1tenth/particle_filter.git
cd ..
sudo apt-get update
rosdep install -r --from-paths src --ignore-src --rosdistro humble -y
colcon build
```
To use it in the simulator, you must comment out the launch of the map_server in the launch file. In the launch/localize_launch.py ​​file, comment out these two lines (Towards the end of the file):
    ld.add_action(nav_lifecycle_node)
    ld.add_action(map_server_node)
Then, in the config/localize.yaml file, change the 'odometry_topic' to '/ego_racecar/odom' and 'range_method' to 'glt
##### Download Rangelibc library
```bash
cd $HOME
# No need to clone this in your workspace, we will only use the python wrapper
git clone https://github.com/f1tenth/range_libc
sudo apt-get install python3-dev cython3
cd range_libc/pywrapper
./compile.sh
sudo TRACE=ON python3 setup.py install
```
##### Fix to visualise "fake" scan

```python
def publish_scan(self, angles, ranges):
# publish the given angels and ranges as a laser scan message
ls = LaserScan()
ls.header.stamp = self.last_stamp
ls.header.frame_id = '/laser'
ls.angle_min = float(np.min(angles))
ls.angle_max = float(np.max(angles))
ls.angle_increment = float(np.abs(angles[0] - angles[1]))
ls.range_min = 0.0
ls.range_max = float(np.max(ranges))
ls.ranges = [float(range) for range in ranges]
self.pub_fake_scan.publish(ls)
```
To be continued
### Planning
#### Global Planning
To create Racelines and centerlines run genertatePaths.py

# Launch
## Bridge
```bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```
## Particle Filter
```bash
ros2 launch particle_filter localize_launch.py
```
## Tests
# Packages
## benchmark_test
## f1tenth_gym_ros
## f110_car_control
## f10_interfaces
## particle_filter
# Topics to subscribe to:
## Perception:
- /scan: scan from lidar
- /ego_racecar/odom: exact position or the car
## Planning
## Control
## Misc
- /ego_crash: True if collision occurs

# Tests
## 
# Create Custom Components
# Simulation to RL
# References


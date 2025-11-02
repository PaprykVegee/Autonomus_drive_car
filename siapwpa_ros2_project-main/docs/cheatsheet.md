# ROS 2 Command Cheat Sheet

## Workspace
### Clone repositories `vcs import`

```
vcs import --recursive < /home/developer/ros2_ws/src/px4.repos src
```

### Update dependencies `rosdep update`

```bash
rosdep update --rosdistro $ROS_DISTRO
```

### Install dependencies `rosdep install`

Dobrą praktyką jest sprawdzanie zależności za każdym razem, gdy dodajemy coś do obszaru roboczego.
Wywołujemy komendę z poziomu folderu, który zawiera `/src` (np. np. `~/ros2_ws`)

```bash
rosdep install -i --from-path src --rosdistro $ROS_DISTRO -y
```

W przypadku powodzenia pojawi się komunikat: `#All required rosdeps installed successfully`

### Build workspace `colcon build`

Budowanie obszaru roboczego odbywa się z poziomu foldera `~/ros2_ws`.

```bash
colcon build --symlink-install
```

Podczas budowania za pomocą `colcon` można wykorzystać następujące flagi ([Więcej informacji](https://colcon.readthedocs.io/en/released/reference/verb/build.html)):

`--packages-up-to` builds the package you want, plus all its dependencies, but not the whole workspace (saves time).  
`--package-select` select the packages with the passed names.
`--symlink-install` saves you from having to rebuild every time you tweak python scripts.  
`--event-handlers` console_direct+ shows console output while building (can otherwise be found in the log directory).  
`--executor` sequential processes the packages one by one instead of using parallelism.  
`--merge-install` With this option most of the paths added to environment variables will be the same, resulting in shorter environment variable values. The disadvantage of using this option is that it doesn’t provide proper isolation between packages.

### Test workspace

W celu przetestowania poprawności zbudowania obszaru roboczego można wywołać:

```bash
colcon test
```

### Clean workspace

Usunięcie folderów `build`, `install`, `log`.

```bash
colcon clean workspace
```

## ROS 2 Package

### Verbs
* `create` - Create a new ROS 2 package.
* `executables` - Output a list of package specific executables.
* `list` - Output a list of available packages.
* `prefix` - Output the prefix path of a package.
* `xml` - Output the information cantained in the package `xml` manifest


**C++ / Python**

```bash
ros2 pkg create --build-type ament_cmake/ament_python --node-name <node_name> <package_name>
```


**Example:**
```bash
ros2 pkg create --build-type ament_python --node-name simple_example simple_example
```
## ROS 2 Launch
```bash
ros2 launch <package> <launch_file_name>
```
**Example:**
```bash
ros2 launch simple_example example.launch.py 
```

## ROS 2 Topic

### Verbs
* `bw` - Display bandwith used by topic.
* `delay` - Display delay of topic from timestamp in header.
* `echo` - Output messages of a given topic to screen.
* `find` - Find topics of a given type.
* `hz` - Display publishing rate of topic.
* `info` - Output information about a given topic.
* `list` - Output list of active topics.
* `pub` - Publish data to a topic.
* `type` - Output topic's type.

**Usage:**
```bash
ros2 topic bw <topic>
ros2 topic echo <topic>
ros2 topic info <topic>
ros2 topic list
ros2 topic pub <topic> <message_type> <data>

```
**Example:**
```bash
ros2 topic pub /camera/trigger std_msgs/msg/Bool "{data: true}" --once
ros2 topic pub /model/vehicle_blue/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 5.0}, angular: {z: 0.5}}"
```

```bash
ros2 topic echo /air_pressure
```

## ROS 2 Run

**Usage:**
```bash
ros2 run <package> <executable>
```

**Example:**
```bash
ros2 run demo_node_cpp talker
```





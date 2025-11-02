# Example of ROS 2 packages in Python

- `package.xml` file containing meta information about the package
- `resource/<package_name>` marker file for the package
- `setup.cfg` is required when a package has executables, so `ros2 run` can find them
- `setup.py` containing instructions for how to install the package
- `<package_name>` - a directory with the same name as your package, used by ROS 2 tools to find your package, contains `__init__.py`

```bash
ros2 pkg create --build-type ament_python --node-name my_node my_package
```

> [!TIP]
> W pliku `package.xml` należy zawsze dodać wszystkie używane biblioteki, jak w przykładzie poniżej:
>
> ```xml
> <exec_depend>rclpy</exec_depend>
> <exec_depend>std_msgs</exec_depend>
> ```
>
> W pliku `setup.py` należy dodać:
>
> ```python
> entry_points={
>        'console_scripts': [
>                '<name_run> = <package_name>.<file_name>:<function_name>',
>        ],
> },
> ```
>
> W pliku `setup.cfg` sprawdzić czy są poprawne dane, jak poniżej:
>
> ```properties
> [develop]
> script_dir=$base/lib/<package_name>
> [install]
> install_scripts=$base/lib/<package_name>
> ```

## How to create simple publisher

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    try:
      rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
      pass
    finally:
      # Destroy the node explicitly
      # (optional - otherwise it will be done automatically
      # when the garbage collector destroys the node object)
      minimal_publisher.destroy_node()
      rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## How to create simple subscriber

```python
import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    try:
      rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
      pass
    finally:
      # Destroy the node explicitly
      # (optional - otherwise it will be done automatically
      # when the garbage collector destroys the node object)
      minimal_subscriber.destroy_node()
      rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## How to create launch file [More info](https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Creating-Launch-Files.html)

```bash
ros2 launch <package_name> <launch_file_name>
```

> [!TIP]
> Unique `namespaces` allow the system to start two similar nodes without `node` name or `topic` name conflicts.

> [!NOTE]
> For packages with `launch` files, it is a good idea to add an `exec_depend` dependency on the `ros2launch` package in your package’s `package.xml`:
>
> ```xml
> <exec_depend>ros2launch</exec_depend>
> ```
> ```python
> setup(
>    # Other parameters ...
>    data_files=[
>        # ... Other data files
>        # Include all launch files.
>        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
>    ]
>)



```python
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    pkg_ros_gz_sim_demos = get_package_share_directory('ros_gz_sim_demos')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-r camera_sensor.sdf'}.items(),
    )

    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(pkg_ros_gz_sim_demos, 'rviz', 'camera.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    # Bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/camera@sensor_msgs/msg/Image@gz.msgs.Image',
                   '/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo'],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Open RViz.'),
        gz_sim,
        bridge,
        rviz
    ])
```

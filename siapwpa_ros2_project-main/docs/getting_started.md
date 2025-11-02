# ROS 2 - Getting Started

The Robot Operating System (ROS) is a set of software libraries and tools for building robot applications. From drivers and state-of-the-art algorithms to powerful developer tools, ROS has the open source tools you need for your next robotics project.

Since ROS was started in 2007, a lot has changed in the robotics and ROS community. The goal of the ROS 2 project is to adapt to these changes, leveraging what is great about ROS 1 and improving what isn’t. - [ROS 2 Humble Documentation](https://docs.ros.org/en/humble/).

## System structure

ROS is a software suite that allows for quick and easy building of autonomous robotic systems. ROS should be considered as a set of tools for creating new solutions or adjusting already existing ones. A major advantage of this system is a great set of drivers and implemented algorithms widely used in robotics.

**Nodes**  
The base unit in ROS is called a node. Nodes are in charge of handling devices or computing algorithms - each node for a separate task. Nodes can communicate with each other using topics or services. ROS software is distributed in packages. A single package is usually developed for performing one type of task and can contain one or multiple nodes.

**Topics**  
In ROS, topic is a data stream used to exchange information between nodes. Topics are used to send frequent messages of one type, such as sensor readouts or information on motor goal speed. Each topic is registered under a unique name and with a defined message type. Nodes can connect to the topic to either publish messages or subscribe to them. For a given topic, one node can not publish and subscribe to it at the same time, but there are no restrictions on the number of different nodes publishing or subscribing.

**Services**  
Communication through services resembles a client-server model. In this mode, one node (the server) registers a service in the system. Then, any other node can ask that service and get a response. In contrast to topics, services allow for two-way communication, as a request can also contain some data.

![ROS 2 Nodes, Topic and Service](images/Nodes-TopicandService.gif)

## ROS 2 Examples

Repozytorium z przykładami dla ROS2 Humble:

```bash
git clone https://github.com/ros/ros_tutorials.git -b humble
```

## Gazebo Harmonic

Gazebo Harmonic posiada wiele gotowych modeli obiektów oraz światów, zarówno po włączeniu symulatora można wybrać przykładowe środowiska, ale także ściągnąć ze strony [Gazebo](https://app.gazebosim.org/dashboard).

Uruchomienie przykładowej symulacji:

```bash
gz sim sonoma.sdf
```

> [!WARNING]
> W przypadku problemów z uruchomieniem Gazebo:
> ```bash 
> ps -ef | grep "gz sim"
> ```
> ```bash 
> pkill -f gz
> ```
> ```bash 
> pkill -f "gz sim"
> ```

## Glossary

- `Underlying communication layer (DDS)` - It provides high-performance, reliable, real-time Underlying communication layer (DDS) data communication and integration capabilities, thereby establishing fundamental support for messaging and service invocation between nodes in ROS2.
- `Node` - Node is the smallest unit of processing running in ROS. It is typically an executable file. Each Node can use topics or services to communicate with other nodes.
- `Message` - The variables of data types such as int, float and Message boolean.
- `Topic` - A one-way asynchronous communication mechanism. By publishing messages to topics or subscribing to topics, the data transmission Topic between nodes can be realized. The topic type is determined by the type of corresponding message.
- `Publishing` - Send data with a message type corresponding to Publishing the topic content.
- `Publishers` - For publishing to take place,the publisher node registers various information such as its topics on Publishers the master node, and transmits messages to subscribing nodes that wish to subscribe.
- `Subscribing` - Receive data with a message type corresponding topic content.
- `Subscribers` - For subscription to take place,the subscribing node registers various information such as its topics on the master node. Subsequently, it receives all messages from publisher nodes that have published topics of interest to this node, via the master node.
- `Services` - A bidirectional synchronous communication mechanism where the service is provided to the client request corresponding to a specific task and service servers gives the service response.
- `Service Servers` - A node taking requests as input, and providing responses as output.
- `Service Clients` - A node taking responding as input, and providing requests as output.
- `URDF file` - A model file describing robot’s entire elements, including link, joint, kinematics parameters, dynamics parameters, visual models and collision detection models.
- `Srv file` - It is stored in the `srv` folder used to define ROS service messages, consisting of two parts: request and respond. The request and respond are separated by the “---” symbol.
- `Msg file` - It is stored in the `msg` folder used to define ROS topic messages.
- `package.xml` - Description of the package attributes, including the package name, version number, authorship and other information.
- `CmakeLists.txt` - Compile the configuration file using Cmake.
- `launch` - Launch files contain system-wide instructions for launching nodes and services required for the robot to operate.



## ROS 2 Package

### C++

- CMakeLists.txt file that describes how to build the code within the package
- `include/<package_name>` directory containing the public headers for the package
- `package.xml` file containing meta information about the package
- `src` directory containing the source code for the package

```bash
ros2 pkg create --build-type ament_cmake --node-name my_node my_package
```

### Python

- `package.xml` file containing meta information about the package
- `resource/<package_name>` marker file for the package
- `setup.cfg` is required when a package has executables, so `ros2 run` can find them
- `setup.py` containing instructions for how to install the package
- `<package_name>` - a directory with the same name as your package, used by ROS 2 tools to find your package, contains `__init__.py`

```bash
ros2 pkg create --build-type ament_python --node-name my_node my_package
```

## List of Tutorials

* [ROS 2 Humble Documentation | ROS Docs](https://docs.ros.org/en/humble/index.html)
* [ROS2 + Gazebo | ROS Docs](https://docs.ros.org/en/humble/Tutorials/Advanced/Simulators/Gazebo/Gazebo.html)
* [Gazebo Sim Docs | Gazebo Sim](https://gazebosim.org/docs/latest/getstarted/)
* [Tutorials | Husarion Docs](https://husarion.com/tutorials/)
* [ROS 2 Humble | Industrial Training](https://industrial-training-master.readthedocs.io/en/humble/)
* [Getting Started with ROS 2 | Ubuntu](https://ubuntu.com/tutorials/getting-started-with-ros-2#1-overview)
* [Tutorials and Courses | The Robotics Back-End](https://roboticsbackend.com/)


## Github Repositories
* [Gazebo Sim Repositories | Gazebo Sim Github](https://github.com/gazebosim)
* [ROS Gazebo Project Template | Gazebo Sim Github](https://github.com/gazebosim/ros_gz_project_template)
* [URDF Tutorial | ROS Github](https://github.com/ros/urdf_tutorial)
* [ROS GZ Bridge | Gazebo Sim Github](https://github.com/gazebosim/ros_gz)
* [ROS GZ Sim Demos | Gazebo Sim Github](https://github.com/gazebosim/ros_gz/tree/ros2/ros_gz_sim_demos)
* [Gazebo Sim with ROS2 | Youtube](https://www.youtube.com/watch?v=DsjJtC8QTQY&ab_channel=TheConstruct)




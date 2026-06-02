*This repository contains all necessary resources for the summer simulations, divided into weekly tasks.*

## WEEK ONE
### The tasks for week one are as follows:
1. Spawn Pupper/SO100 in MuJoCo, look into .mjcf format which is different from urdf/xacro.
2. Check mass, joint and actuator limits, prop sensor config.
3. Write a function to send torque cmd and check joint movement.
4. Write a function to read and log prop sensor data to the terminal.
5. Finally, add sliders to control joint angles and observe if it moves correctly.

### Getting Started with the setup

1. Clone the repo
    > git clone --recurse-submodules https://github.com/AviralxD/the-sims.git

    You can also [fork the repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo), in that case be sure to ``` git fetch ``` the latest changes.

2. Download mujoco and try loading some examples in the default viewer.
    > [mujoco.org/download](https://mujoco.org/download)

3. Convert the urdf to mjcf, you can use [this repo](https://github.com/kscalelabs/urdf2mjcf). Try to mess around with the generated xml to understand how the mjcf format works. 

    Tip: joints and actuators need physics like damping, friction, limits etc; do look into contype and conaffinity.

#### Once MuJoCo is setup and the bot is spawned, proceed with tasks 2 - 5

Use this code as a template to build upon.

```
import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("path_of_model/model.xml")
data = mujoco.MjData(model)

for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"Joint {i}: {name}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():   
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
```

### Resources:

- [mujoco docs](https://mujoco.readthedocs.io/en/stable/overview.html)
- [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/): compilation of mjcf files for various robots
- [urdf2mjcf](https://github.com/kscalelabs/urdf2mjcf): urdf to mujoco converter
- [mujoco_ros_pkgs](https://github.com/ubi-agni/mujoco_ros_pkgs.git): ROS1 wrapper for mujoco

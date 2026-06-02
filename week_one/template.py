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
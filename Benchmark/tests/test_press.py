import mujoco
import numpy as np
import mujoco.viewer as viewer
import time

# Load the XML model (assuming the XML above is saved as a string named 'xml_string')
xml_string = """
<mujoco model="pressure_sensor_example">
  <compiler angle="degree"/>
  <option gravity="0 -3 -9.81"/>

  <worldbody>
    <light pos="0 0 1" dir="0 0 -1" directional="true" diffuse="0.5 0.8 0.1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.9 0.8 1"/>

    <body name="falling_box" pos="0 0 0.5">
		<freejoint/>
		<geom name="box_geom" type="box" size="0.1 0.1 0.1" rgba="0.2 0.6 0.8 1"/>
		
		<site name="bottom_sensor_site" type="box" size="0.101 0.101 0.01" pos="0 0 -0.1" rgba="1 0 0 0.5"/>
	</body>
  </worldbody>

  <sensor>
    <touch name="box_bottom_pressure" site="bottom_sensor_site" cutoff="10"/>
  </sensor>
</mujoco>
"""

'''
<site name="bottom_sensor_site" type="box" size="0.09 0.09 0.012" pos="0 0 -0.1" rgba="1 0 0 0.5"/>
<site name="bottom_sensor_site" type="box" size="0.105 0.105 0.01" pos="0 0 -0.1" rgba="1 0 0 0.5"/>
'''

model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)
# 1. Use launch_passive to run the viewer in the background
with viewer.launch_passive(model, data) as viewer:
    print("Starting loop...")
    time.sleep(3)
    
    # Simulation loop
    for step in range(3350):
        mujoco.mj_step(model, data)
        
        # 2. Sync the viewer to update the visual state
        viewer.sync()
        
        # Read the sensor data
        pressure_force = data.sensor('box_bottom_pressure').data[0]
        
        if np.abs(pressure_force) > 0:
            print(f"Step {step} | Contact detected! Normal force/pressure: {pressure_force:.2f} N, {data.sensor('box_bottom_pressure').data}")
            
        # 3. Sleep for the duration of the physics timestep to visualize in roughly real-time
        time.sleep(model.opt.timestep)
        #time.sleep(1)
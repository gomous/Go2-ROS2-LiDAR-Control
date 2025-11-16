#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Walk These Ways Controller - Simulation Only
"""

# # Test In Simulation

from Go2Py.robot.fsm import FSM
from Go2Py.robot.remote import KeyboardRemote
from Go2Py.robot.safety import SafetyHypervisor
from Go2Py.sim.mujoco import Go2Sim
from Go2Py.control.walk_these_ways import *

robot = Go2Sim()

map = np.zeros((1200, 1200))
map[:200, :200] = 255
robot.updateHeightMap(map)

remote = KeyboardRemote()
robot.sitDownReset()
safety_hypervisor = SafetyHypervisor(robot)

class walkTheseWaysController:
    def __init__(self, robot, remote, checkpoint):
        self.remote = remote
        self.robot = robot
        self.cfg = loadParameters(checkpoint)
        self.policy = Policy(checkpoint)
        self.command_profile = CommandInterface()
        self.agent = WalkTheseWaysAgent(self.cfg, self.command_profile, self.robot)
        self.agent = HistoryWrapper(self.agent)
        self.hist_data = {}
        self.init()

    def init(self):
        self.obs = self.agent.reset()
        self.policy_info = {}
        self.command_profile.yaw_vel_cmd = 0.0
        self.command_profile.x_vel_cmd = 0.0
        self.command_profile.y_vel_cmd = 0.0
        self.command_profile.stance_width_cmd=0.25
        self.command_profile.footswing_height_cmd=0.08
        self.command_profile.step_frequency_cmd = 3.0
        self.command_profile.bodyHeight = 0.00

    def update(self, robot, remote):
        action = self.policy(self.obs, self.policy_info)
        self.obs, self.ret, self.done, self.info = self.agent.step(action)
        
        for key, value in self.info.items():
            if key in self.hist_data:
                self.hist_data[key].append(value)
            else:
                self.hist_data[key] = [value]

checkpoint = "../Go2Py/assets/checkpoints/walk_these_ways/"
controller = walkTheseWaysController(robot, remote, checkpoint)

# Set initial gait parameters
controller.command_profile.pitch_cmd=0.0
controller.command_profile.body_height_cmd=0.0
controller.command_profile.footswing_height_cmd=0.08
controller.command_profile.roll_cmd=0.0
controller.command_profile.stance_width_cmd=0.2
controller.command_profile.x_vel_cmd=0.3  # Forward velocity
controller.command_profile.y_vel_cmd=0.0
controller.command_profile.setGaitType("trotting")

# Create FSM
fsm = FSM(robot, remote, safety_hypervisor, user_controller_callback=controller.update)

# Pressing `u` on the keyboard will make the robot stand up. 
# After the robot is on its feet, pressing `s` will hand over control to the RL policy.
# Pressing `u` again will lock it in standing mode.
# Pressing `u` one more time will command the robot to sit down.

# To change velocity commands while running:
# controller.command_profile.x_vel_cmd = 0.5  # forward/backward
# controller.command_profile.y_vel_cmd = 0.2  # left/right
# controller.command_profile.yaw_vel_cmd = 0.3  # rotation

# When done, close everything
# fsm.close()
# robot.close()
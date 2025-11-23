#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

from enum import Enum

import carla
import os
import moviepy
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from srunner.scenariomanager.timer import GameTime

from leaderboard.utils.route_manipulation import downsample_route
from leaderboard.envs.sensor_interface import SensorInterface
import numpy as np


def get_entry_point():
    return "AutonomousAgent"


from leaderboard.autoagents.autonomous_agent import Track


class AutonomousAgent(object):
    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, carla_host, carla_port, debug=False):
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()

        self.wallclock_t0 = None

        self.video_path = None

        self.render_res_w = None
        self.render_res_h = None

        self.display_res_w = None
        self.display_res_h = None

        self.obs_res_w = None
        self.obs_res_h = None
        self.obs_res_c = None

        self.sensors_list = None

        self.steps = 0

        self.frames_to_record = []  # frames to record for mp4 video

        self.raw_files = None

    def setup(self, args):
        self.video_path = args.video_path
        self.raw_files = args.raw_files
        if self.raw_files != "":
            os.makedirs(self.raw_files, exist_ok=True)

        self.render_res_w, self.render_res_h = [
            int(x) for x in args.render_res.split("x")
        ]
        if args.display_res:
            self.display_res_w, self.display_res_h = [
                int(x) for x in args.display_res.split("x")
            ]

        if args.obs_res:
            self.obs_res_w, self.obs_res_h, self.obs_res_c = [
                int(x) for x in args.obs_res.split("x")
            ]

        self.sensors_list = [
            {
                "type": "sensor.camera.rgb",
                "x": 0.7,
                "y": 0.0,
                "z": 1.60,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": self.render_res_w,
                "height": self.render_res_h,
                "fov": args.fov,
                "id": "Center",
            },
        ]

        self.fps = args.frame_rate

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:
        """

        return self.sensors_list

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        if self.steps % 10 == 0:
            print(
                "frame:",
                input_data["Center"][0],
                "timestamp:",
                timestamp,
                "steps:",
                self.steps,
            )

        frame_to_record = input_data["Center"][1][:, :, -2::-1]
        if self.video_path:
            self.frames_to_record.append(frame_to_record)

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        if self.video_path:
            print("Recording video...")
            os.makedirs(os.path.dirname(self.video_path), exist_ok=True)

            if self.raw_files != "":
                np.save(
                    self.raw_files + "/frames_to_record.npy",
                    np.array(self.frames_to_record),
                )

            clip = ImageSequenceClip(self.frames_to_record, fps=self.fps)
            clip.write_videofile(self.video_path, codec="libx264")
            self.frames_to_record = []

    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        input_data = self.sensor_interface.get_data(GameTime.get_frame())

        timestamp = GameTime.get_time()

        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()
        sim_ratio = 0 if wallclock_diff == 0 else timestamp / wallclock_diff

        # if self.steps % 10 == 0:
        #     print('=== [Agent] -- Wallclock = {} -- System time = {} -- Game time = {} -- Ratio = {}x'.format(
        #         str(wallclock)[:-3], format(wallclock_diff, '.3f'), format(timestamp, '.3f'), format(sim_ratio, '.3f')))

        control = self.run_step(input_data, timestamp)
        control.manual_gear_shift = False

        self.steps += 1

        return control

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 200)
        self._global_plan_world_coord = [
            (global_plan_world_coord[x][0], global_plan_world_coord[x][1])
            for x in ds_ids
        ]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]


def control_to_vector(control):
    """
    Convert the control to a vector
    """
    # print("Control is:", 'throttle:', control.throttle, 'steer:', control.steer, 'brake:', control.brake, 'hand_brake:', control.hand_brake, 'reverse:', control.reverse, 'manual_gear_shift:', control.manual_gear_shift, 'gear:', control.gear)
    v = np.array(
        [
            control.throttle,
            control.steer,
            control.brake,
            control.hand_brake,
            control.reverse,
            control.manual_gear_shift,
            control.gear,
        ],
        dtype=np.float32,
    )
    return v


def vector_to_control(vector):
    """
    Convert the vector to a control
    """
    # vector = np.clip(vector, 0.0, 1.0)

    control = carla.VehicleControl()
    control.throttle = float(np.clip(vector[0], 0.0, 1.0))
    control.steer = float(np.clip(vector[1], -1.0, 1.0))
    control.brake = float(vector[2] > 0.8)
    control.hand_brake = bool(vector[3] > 0.5)
    control.reverse = bool(vector[4] > 0.5)
    control.manual_gear_shift = bool(vector[5] > 0.5)
    control.gear = int(vector[6])
    # print("Control is:", 'throttle:', control.throttle, 'steer:', control.steer, 'brake:', control.brake, 'hand_brake:', control.hand_brake, 'reverse:', control.reverse, 'manual_gear_shift:', control.manual_gear_shift, 'gear:', control.gear)
    return control


def noop_control():
    """
    Return a no-op control
    """
    control = carla.VehicleControl()
    control.throttle = 0.0
    control.steer = 0.0
    control.brake = 1.0
    control.hand_brake = False
    control.reverse = False
    control.manual_gear_shift = False
    control.gear = 0
    return control

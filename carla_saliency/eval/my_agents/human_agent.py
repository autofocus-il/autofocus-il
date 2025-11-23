#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""
import torch
import numpy as np
import cv2

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

import carla

from my_agents.autonomous_agent import (
    AutonomousAgent,
    control_to_vector,
    vector_to_control,
)
from sensor import GazepointClient


class HumanInterface(object):
    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(self, display_width, display_height, enable_display):
        self.display_width = display_width
        self.display_height = display_height
        self.enable_display = enable_display
        self._surface = None

        # self.gaze_detector = GazepointClient()
        # self.gaze_data = [0.5, 0.5, 0]
        # self.step_number = 0
        if enable_display:
            pygame.init()
            pygame.font.init()
            self._clock = pygame.time.Clock()
            self._display = pygame.display.set_mode(
                (self.display_width, self.display_height), pygame.FULLSCREEN
            )
            pygame.display.set_caption("Human Agent")

            # check if os is windows
            import os

            if os.name == "nt":
                # if windows, play a sound
                import winsound

                winsound.Beep(440, 1000)

    def run_interface(self, image):
        """
        Run the GUI
        """
        if not self.enable_display:
            return

        self._surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

        # Display image
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def set_black_screen(self):
        """Set the surface to black"""
        if not self.enable_display:
            return

        black_array = np.zeros([self.display_width, self.display_height])
        self._surface = pygame.surfarray.make_surface(black_array)
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def _quit(self):
        if not self.enable_display:
            return

        pygame.quit()


class HumanAgent(AutonomousAgent):
    """
    Human agent to control the ego vehicle via keyboard
    """

    def setup(self, args):
        """
        Setup the agent parameters
        """
        super(HumanAgent, self).setup(args)

        self.dataset_path = args.dataset_path
        self.enable_gaze = args.enable_gaze
        self.collect_obs = args.collect_obs
        self.enable_display = args.enable_display
        self.display_gaze = args.display_gaze
        self.mode = args.human_mode

        self.gaze_source = args.gaze_source

        self.controller_type = args.controller

        assert self.controller_type in [
            "keyboard",
            "joystick",
        ], "Invalid controller type"

        assert self.mode in ["replay", "collect"], "Invalid mode"
        assert (
            self.mode != "collect" or self.enable_gaze
        ), "Collecting gaze data is required for the collect mode"
        assert (
            self.mode != "collect" or self.enable_display
        ), "Display is required for the collect mode"
        assert (
            self.mode != "replay" or self.dataset_path
        ), "Dataset path is required for the replay mode"

        self.agent_engaged = False
        self.current_control = None

        self._hic = HumanInterface(
            self.display_res_w, self.display_res_h, self.enable_display
        )

        if self.mode == "collect":
            self.actions = []  # actions
            self.timestaps = []  # timestamps
            self._controller = (
                KeyboardControl()
                if self.controller_type == "keyboard"
                else JoystickControl()
            )
            self._prev_timestamp = 0

            if self.enable_gaze:
                self.gaze_data = []
                if self.gaze_source == "human":
                    self.gaze_sensor = GazepointClient(host=args.gaze_address)

                self.last_valid_gaze = (0.5, 0.5, 0)
        else:
            d = torch.load(self.dataset_path + "actions.pt")
            self.actions = d["actions"]
            self.timestaps = d["timestamps"]

            if self.enable_gaze:
                self.gaze_data = torch.load(self.dataset_path + "gaze.pt")

        if self.collect_obs:
            self.observations = []  # observations

        self._clock = pygame.time.Clock()

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """

        self._clock.tick_busy_loop(self.fps)

        image_center = input_data["Center"][1][:, :, -2::-1]

        if self.collect_obs:
            obs = input_data["Center"][1][:, :, -2::-1]
            if self.obs_res_c == 1:
                obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)

            if (
                self.obs_res_w != self.render_res_w
                or self.obs_res_h != self.render_res_h
            ):
                obs = cv2.resize(obs, (self.obs_res_w, self.obs_res_h))

            self.observations.append(obs)

        if self.enable_gaze:
            if self.mode == "collect":
                if self.gaze_source == "human":
                    current_gaze_info = self.gaze_sensor.receive_data()
                elif self.gaze_source == "dummy":
                    new_x = self.last_valid_gaze[0] + 0.02
                    new_y = self.last_valid_gaze[1] + 0.02
                    if new_x > 1.0:
                        new_x = 0.0
                    if new_y > 1.0:
                        new_y = 0.0

                    current_gaze_info = [new_x, new_y]
                elif self.gaze_source == "center":
                    current_gaze_info = [0.5, 0.5]
                else:
                    raise ValueError("Invalid gaze source")

                if (
                    current_gaze_info != [None, None]
                    and current_gaze_info != [0, 0]
                    and current_gaze_info != [0.0, 1.0]
                    and current_gaze_info != [1.0, 0.0]
                    and current_gaze_info != [1.0, 1.0]
                ):
                    self.last_valid_gaze = (
                        current_gaze_info[0],
                        current_gaze_info[1],
                        self.steps,
                    )

                self.gaze_data.append(self.last_valid_gaze)
            else:
                self.last_valid_gaze = self.gaze_data[self.steps]

            if self.display_gaze:
                gaze = np.array(self.last_valid_gaze)
                gaze[0] = gaze[0] * self.render_res_w
                gaze[1] = gaze[1] * self.render_res_h
                # Add gaze marker here
                pix_square_size = 10 / 1920 * self.render_res_w
                # Add circle here
                image_center = cv2.circle(
                    image_center.copy(),
                    (int(gaze[0]), int(gaze[1])),
                    int(pix_square_size),
                    (190, 0, 190),
                    -1,
                )

        self.frames_to_record.append(image_center)

        if self.enable_display:
            if (
                self.display_res_w != self.render_res_w
                or self.display_res_h != self.render_res_h
            ):
                image_center = cv2.resize(
                    image_center, (self.display_res_w, self.display_res_h)
                )

        self.agent_engaged = True
        self._hic.run_interface(image_center)

        if self.mode == "collect":
            control = self._controller.parse_events(timestamp - self._prev_timestamp)
            self.actions.append(control_to_vector(control))
            self.timestaps.append(timestamp)
            self._prev_timestamp = timestamp

        else:
            control = vector_to_control(self.actions[self.steps])

        self.steps += 1
        return control

    def destroy(self):
        """
        Cleanup
        """
        self._hic.set_black_screen()
        self._hic._quit = True

        if self.dataset_path:
            self.actions = np.stack(self.actions)
            torch.save(
                {"actions": self.actions, "timestamps": self.timestaps},
                self.dataset_path + "actions.pt",
            )

            if self.collect_obs:
                self.observations = np.stack(self.observations)
                torch.save(self.observations, self.dataset_path + "observations.pt")

            if self.enable_gaze:
                torch.save(self.gaze_data, self.dataset_path + "gaze.pt")

        super(HumanAgent, self).destroy()


class JoystickControl(object):
    def __init__(self):
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("No joystick detected!")
            exit()

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        print(f"Controller Name: {self.joystick.get_name()}")

    def get_current_controller_state(self):
        pygame.event.pump()
        V = []
        # Buttons
        for button in range(self.joystick.get_numbuttons()):
            button_state = self.joystick.get_button(button)
            V.append(button_state)

        # Axes (joysticks)
        for axis in range(self.joystick.get_numaxes()):
            axis_value = self.joystick.get_axis(axis)
            V.append(axis_value)

        return V

    def parse_events(self, timestamp):
        """
        Parse the joystick events and set the vehicle controls accordingly
        """
        self._parse_vehicle_keys(self.get_current_controller_state(), timestamp * 1000)

        return self._control

    def _parse_vehicle_keys(self, inputs, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """
        x = inputs[16]
        y = -inputs[19]
        self._control.throttle = 0.8 * y if y > 0 else 0
        self._control.brake = -y if y <= 0 else 0

        new_steer = 0.99 * self._steer_cache + 0.01 * x if abs(x) > 0.1 else 0
        self._control.steer = new_steer
        self._control.hand_brake = False
        self._control.reverse = False
        self._control.gear = 1

        self._steer_cache = new_steer

    def __del__(self):
        # Get ready to log user commands
        pass


class KeyboardControl(object):
    """
    Keyboard control for the human agent
    """

    def __init__(self):
        """
        Init
        """
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0

    def parse_events(self, timestamp):
        """
        Parse the keyboard events and set the vehicle controls accordingly
        """
        self._parse_vehicle_keys(pygame.key.get_pressed(), timestamp * 1000)

        return self._control

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYUP:
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                    self._control.reverse = self._control.gear < 0

        if keys[K_UP] or keys[K_w]:
            self._control.throttle = 0.8
        else:
            self._control.throttle = 0.0

        steer_increment = 3e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def __del__(self):
        # Get ready to log user commands
        pass

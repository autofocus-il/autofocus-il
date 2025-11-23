import argparse
import json
import traceback
from argparse import RawTextHelpFormatter

from my_agents.bc_agent import BCAgent
from my_agents.human_agent import HumanAgent

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import (
    AgentError,
    validate_sensor_configuration,
)
from leaderboard.utils.statistics_manager import StatisticsManager, FAILURE_MESSAGES
from leaderboard.utils.route_indexer import RouteIndexer

sensors_to_icons = {
    "sensor.camera.rgb": "carla_camera",
    "sensor.lidar.ray_cast": "carla_lidar",
    "sensor.other.radar": "carla_radar",
    "sensor.other.gnss": "carla_gnss",
    "sensor.other.imu": "carla_imu",
    "sensor.opendrive_map": "carla_opendrive_map",
    "sensor.speedometer": "carla_speedometer",
}

import carla
import pkg_resources
from distutils.version import LooseVersion
import os
import importlib
import signal
import sys


class LeaderboardEvaluator(object):
    """
    Main class of the Leaderboard. Everything is handled from here,
    from parsing the given files, to preparing the simulation, to running the route.
    """

    def __init__(self, args):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.world = None
        self.manager = None
        self.sensors = None
        self.sensors_initialized = False
        self.sensor_icons = []
        self.agent_instance = None
        self.route_scenario = None

        self.frame_rate = args.frame_rate
        self.client_timeout = 1 * 60  # in seconds
        self.timeout = 300  # in seconds

        self.statistics_manager = StatisticsManager(
            args.checkpoint, args.debug_checkpoint
        )

        # Setup the simulation
        self.client, self.traffic_manager = self._setup_simulation(args)

        dist = pkg_resources.get_distribution("carla")
        if dist.version != "leaderboard":
            if LooseVersion(dist.version) < LooseVersion("0.9.10"):
                raise ImportError(
                    "CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(
                        dist
                    )
                )

        # Create the ScenarioManager
        self.manager = ScenarioManager(
            self.timeout, self.statistics_manager, args.debug
        )

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Prepare the agent timer
        self._agent_watchdog = None
        signal.signal(signal.SIGINT, self._signal_handler)

        self._client_timed_out = False

    def _setup_simulation(self, args):
        """
        Prepares the simulation by getting the client, and setting up the world and traffic manager settings
        """
        client = carla.Client(args.host, args.port)

        client.set_timeout(self.client_timeout)

        settings = carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=1.0 / self.frame_rate,
            deterministic_ragdolls=True,
        )  # spectator_as_ego = False
        client.get_world().apply_settings(settings)

        traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_hybrid_physics_mode(True)

        return client, traffic_manager

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt.
        Either the agent initialization watchdog is triggered, or the runtime one at scenario manager
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError(
                "Timeout: Agent took longer than {}s to setup".format(
                    self.client_timeout
                )
            )
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """
        if hasattr(self, "manager") and self.manager:
            del self.manager
        if hasattr(self, "world") and self.world:
            del self.world

    def _get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._agent_watchdog:
            return self._agent_watchdog.get_status()
        return False

    def _cleanup(self):
        """
        Remove and destroy all actors
        """
        CarlaDataProvider.cleanup()

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        try:
            results = {}
            if self.agent_instance:
                self.agent_instance.destroy()
                self.agent_instance = None
        except Exception as e:
            print("\n\033[91mFailed to stop the agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

        if self.route_scenario:
            self.route_scenario.remove_all_actors()
            self.route_scenario = None
            if self.statistics_manager:
                self.statistics_manager.remove_scenario()

        if self.manager:
            self._client_timed_out = not self.manager.get_running_status()
            self.manager.cleanup()

        # Make sure no sensors are left streaming
        alive_sensors = self.world.get_actors().filter("*sensor*")
        for sensor in alive_sensors:
            sensor.stop()
            sensor.destroy()

    def _reset_world_settings(self):
        """
        Changes the modified world settings back to asynchronous
        """
        # Has simulation failed?
        if self.world and self.manager and not self._client_timed_out:
            # Reset to asynchronous mode
            self.world.tick()  # TODO: Make sure all scenario actors have been destroyed
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            settings.deterministic_ragdolls = False
            settings.spectator_as_ego = True
            self.world.apply_settings(settings)

            # Make the TM back to async
            self.traffic_manager.set_synchronous_mode(False)
            self.traffic_manager.set_hybrid_physics_mode(False)

    def _load_and_wait_for_world(self, args, town):
        """
        Load a new CARLA world without changing the settings and provide data to CarlaDataProvider
        """
        print("*************** Town is:", town)
        self.world = self.client.load_world(town, reset_settings=False)

        # Large Map settings are always reset, for some reason
        settings = self.world.get_settings()
        settings.tile_stream_distance = 650
        settings.actor_active_distance = 650
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_traffic_manager_port(args.traffic_manager_port)
        CarlaDataProvider.set_world(self.world)

        # This must be here so that all route repetitions use the same 'unmodified' seed
        self.traffic_manager.set_random_device_seed(args.seed)

        CarlaDataProvider._rng = random.RandomState(seed=args.seed)

        # Wait for the world to be ready
        self.world.tick()

        map_name = CarlaDataProvider.get_map().name.split("/")[-1]
        if map_name != town:
            raise Exception(
                "The CARLA server uses the wrong map!"
                " This scenario requires the use of map {}".format(town)
            )

    def _register_statistics(self, route_index, entry_status, crash_message=""):
        """
        Computes and saves the route statistics
        """
        print("\033[1m> Registering the route statistics\033[0m")
        self.statistics_manager.save_entry_status(entry_status)
        self.statistics_manager.compute_route_statistics(
            route_index,
            self.manager.scenario_duration_system,
            self.manager.scenario_duration_game,
            crash_message,
        )

    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"

        print(
            "\n\033[1m========= Preparing {} (repetition {}) =========\033[0m".format(
                config.name, config.repetition_index
            )
        )

        # Prepare the statistics of the route
        route_name = f"{config.name}_rep{config.repetition_index}"
        self.statistics_manager.create_route_data(route_name, config.index)

        print("\033[1m> Loading the world\033[0m")

        # Load the world and the scenario
        try:
            self._load_and_wait_for_world(args, config.town)
            self.route_scenario = RouteScenario(
                world=self.world, config=config, debug_mode=args.debug
            )
            self.statistics_manager.set_scenario(self.route_scenario)

        except Exception:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return True

        print("\033[1m> Setting up the agent\033[0m")

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            if args.video_path == "auto":
                if args.agent == "human":
                    args.video_path = (
                        f"Vids/{args.agent}/route_{args.routes_id}/seed_{args.seed}.mp4"
                    )
                elif args.agent == "BC":
                    args.video_path = f"{args.params_path}/route_{args.routes_id}/seed_{args.seed}.mp4"

                os.makedirs(os.path.dirname(args.video_path), exist_ok=True)
            self._agent_watchdog = Watchdog(self.timeout)
            self._agent_watchdog.start()

            AGENT_CLASS = {"human": HumanAgent, "BC": BCAgent, "npc": None}[args.agent]

            self.agent_instance = AGENT_CLASS(args.host, args.port, args.debug)
            self.agent_instance.set_global_plan(
                self.route_scenario.gps_route, self.route_scenario.route
            )
            self.agent_instance.setup(args)

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [
                    sensors_to_icons[sensor["type"]] for sensor in self.sensors
                ]
                self.statistics_manager.save_sensors(self.sensor_icons)
                self.statistics_manager.write_statistics()

                self.sensors_initialized = True

            self._agent_watchdog.stop()
            self._agent_watchdog = None

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print(f"{e}\033[0m\n")

            entry_status, crash_message = FAILURE_MESSAGES["Sensors"]
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return True

        except Exception:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Agent_init"]
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return False

        print("\033[1m> Running the route\033[0m")

        # Run the scenario
        try:
            # Load scenario and run it
            self.manager.load_scenario(
                self.route_scenario,
                self.agent_instance,
                config.index,
                config.repetition_index,
            )
            self.manager.run_scenario()

        except AgentError:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Agent_runtime"]

        except Exception:
            print("\n\033[91mError during the simulation:")
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]

        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()
            self._register_statistics(config.index, entry_status, crash_message)

            self._cleanup()

        except Exception:
            print(
                "\n\033[91mFailed to stop the scenario, the statistics might be empty:"
            )
            print(f"\n{traceback.format_exc()}\033[0m")

            _, crash_message = FAILURE_MESSAGES["Simulation"]

        # If the simulation crashed, stop the leaderboard, for the rest, move to the next route
        return crash_message == "Simulation crashed"

    def run(self, args):
        """
        Run the challenge mode
        """

        if args.agent == "human":
            args.dataset_path = (
                f"dataset/bench2drive220/route_{args.routes_id}/seed_{args.seed}/"
            )
            os.makedirs(args.dataset_path, exist_ok=True)
        elif args.agent == "BC":
            args.stats_path = (
                args.params_path + f"/route_{args.routes_id}/seed_{args.seed}/"
            )
            os.makedirs(args.stats_path, exist_ok=True)
        else:
            raise ValueError("Unknown agent type")

        route_indexer = RouteIndexer(args.routes, 1, str(args.routes_id))

        self.statistics_manager.clear_records()
        self.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
        self.statistics_manager.write_statistics()

        crashed = False
        while (
            route_indexer.peek() and not crashed
        ):  # this is expected to run a single time since repetitions is set to 1 and routes_id is single int
            # Run the scenario
            config = route_indexer.get_next_config()
            crashed = self._load_and_run_scenario(args, config)

            # Save the progress and write the route statistics
            self.statistics_manager.save_progress(
                route_indexer.index, route_indexer.total
            )
            self.statistics_manager.write_statistics()

        # Go back to asynchronous mode
        self._reset_world_settings()

        results = None
        if not crashed:
            # Save global statistics
            print("\033[1m> Registering the global statistics\033[0m")
            self.statistics_manager.compute_global_statistics()
            results = self.statistics_manager.validate_and_write_statistics(
                self.sensors_initialized, crashed
            )

            if args.agent == "human":
                # write results to a json file
                file_path = args.dataset_path + "stats.json"
                with open(file_path, "w") as f:
                    json.dump(results, f)
            elif args.agent == "BC":
                # write results to a json file
                file_path = args.stats_path + "stats.json"
                with open(file_path, "w") as f:
                    json.dump(results, f)

        return crashed, results


def main(args):
    leaderboard_evaluator = LeaderboardEvaluator(args)
    crashed, results = leaderboard_evaluator.run(args)

    del leaderboard_evaluator

    # if crashed:
    #     sys.exit(-1)
    # else:
    #     sys.exit(0)

    return crashed, results


def get_args():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--host", default="localhost", help="IP of the host server (default: localhost)"
    )
    parser.add_argument("--port", default=6100, type=int, help="TCP port to listen to")
    parser.add_argument(
        "--traffic-manager-port",
        default=3100,
        type=int,
        help="Port to use for the TrafficManager (default: 8000)",
    )

    parser.add_argument(
        "--frame_rate",
        default=20.0,
        type=float,
        help="The frame rate of the simulation (default: 20.0)",
    )

    # simulation setup
    parser.add_argument(
        "--routes",
        default="routes/bench2drive220.xml",
        help="Name of the routes file to be executed.",
    )
    parser.add_argument(
        "--routes-id", default=3100, type=int, help="Execute a specific set of routes"
    )
    parser.add_argument(
        "--seed",
        default=199,
        type=int,
        help="Seed used by the TrafficManager (default: 0)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/simulation_results.json",
        help="Path to checkpoint used for saving statistics and resuming",
    )
    parser.add_argument(
        "--debug-checkpoint",
        type=str,
        default="checkpoints/live_results.txt",
        help="Path to checkpoint used for saving live results",
    )
    parser.add_argument(
        "--track", type=str, default="SENSORS", help="Participation track: SENSORS, MAP"
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="auto",
        help="Record the simulation to a given file",
    )
    parser.add_argument(
        "--render_res",
        type=str,
        default="1920x1080",
        help="Resolution of the camera sensor",
    )
    parser.add_argument(
        "--display_res",
        type=str,
        default="1920x1080",
        help="Display resolution to the user",
    )
    parser.add_argument(
        "--obs_res",
        type=str,
        default="320x180x3",
        help="Resolution of the input to the model",
    )
    parser.add_argument(
        "--fov", type=int, default=60, help="Field of view of the camera sensor"
    )

    parser.add_argument(
        "--agent",
        type=str,
        default="human",
        choices=["human", "BC"],
        help="Select the agent to run",
    )
    parser.add_argument(
        "--human_mode",
        type=str,
        default="collect",
        choices=["collect", "replay"],
        help="Select the human mode",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="keyboard",
        choices=["keyboard", "joystick"],
        help="Select the controller",
    )
    parser.add_argument(
        "--collect_obs", type=bool, default=True, help="Collect observations"
    )
    parser.add_argument(
        "--enable_gaze", type=bool, default=True, help="Enable gaze recording"
    )
    parser.add_argument(
        "--gaze_source",
        type=str,
        default="dummy",
        choices=["dummy", "center", "human"],
        help="Gaze source",
    )
    parser.add_argument(
        "--enable_display", type=bool, default=True, help="Enable display"
    )
    parser.add_argument(
        "--display_gaze", type=bool, default=True, help="Display gaze on screen"
    )
    parser.add_argument(
        "--gaze_address", type=str, default="localhost", help="GazePoint Sensor Address"
    )

    parser.add_argument(
        "--params_path", type=str, default="", help="Path to params for the BC agent"
    )

    parser.add_argument(
        "--raw_files", type=str, default="", help="Path to save raw frames"
    )

    parser.add_argument(
        "--debug", type=int, default=2, help="Debug mode in 0, 1, 2", choices=[0, 1, 2]
    )

    # Confounded overlay (draw action indicators on recorded frames)
    parser.add_argument(
        "--confounded",
        action="store_true",
        help="Overlay action indicators onto frames during evaluation",
    )
    parser.add_argument(
        "--confounded_config",
        type=str,
        default="carla_saliency/configs/confounded_render.yaml",
        help="YAML config for confounded overlay rendering",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    main(args)

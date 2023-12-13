import numpy as np
from gym.envs.registration import register
import math
from typing import Tuple

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.dynamics import ControlledBicycleVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.objects import Landmark

class LaneChnageMARL(AbstractEnv):

    n_a = 2
    n_s = 26
    
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "ContinuousAction",
                        "lateral": False,
                        "longitudinal": True,
                        "dynamical": True,
                    },
                },
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {"type": "KinameticObsExt"},
                },
                "lanes_count": 2,
                "min_speeds": [27.77, 16.66],
                "max_speeds": [33.33, 27.77],
                "controlled_vehicles": 5,
                "safety_guarantee": False,
                "action_masking": False,
                "target_lane": 0,
                "initial_lane_id": 1,
                "length": 300,
                "screen_width": 1200,
                "screen_height": 100,
                "centering_position": [0.3, 0.5],
                # "scaling": 7,
                "scaling": 4,
                "simulation_frequency": 15,  # [Hz]
                "duration": 20,  # time step
                "policy_frequency": 5,  # [Hz]
                "reward_speed_range": [10, 30],

            }
        )
        return config

    def _reset(self, num_CAV=0) -> None:
        self._create_road()
        self._create_goal()
        self._create_vehicles()
        # self.action_is_safe = True
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        return sum(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        ) / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: ControlledBicycleVehicle) -> float:
        """
        The first vehicle is rewarded for 
            - moving towards the middle of target lane,
        All the vehicles are rewarded for
            - moving in the middle of the lane
        :param action: the action performed
        :return: the reward of the state-action transition
        """

        r_lateral = 0
        r_lon = 0
        r_lane = 0

        if vehicle.goal is not None and (vehicle.lane_index[2] != self.config["target_lane"]):
            # penalty for staying in wrong lane 
            r_lateral = r_lon = -0.5

            # one time reward for reaching the goal
            if vehicle.goal is not None and vehicle.goal.hit:
                r_lateral = r_lon = 100
                vehicle.goal = None

            # penalty for not reaching the target lane
            if self._agent_is_terminal(vehicle):
                r_lateral = r_lon = -100
        else:
            # one time reward for reaching the goal
            if vehicle.goal is not None and vehicle.goal.hit:
                r_lateral = r_lon = 100
                vehicle.goal = None
            
            # reward for lane following
            lon, lat = vehicle.lane.local_coordinates(vehicle.position)
            r_lateral = 1 if vehicle.heading == 0 and lat == 0 else 0

            cur_lane = vehicle.lane

            if cur_lane.min_speed <= vehicle.speed <= cur_lane.speed_limit:
                r_lon = 2 - 0.01 * (cur_lane.speed_limit - vehicle.speed)
            elif -20 <= (cur_lane.speed_limit - vehicle.speed) <= 0:
                r_lon = 0.5 + 0.01 * (cur_lane.speed_limit - vehicle.speed)
            elif -40 <= (vehicle.speed - cur_lane.min_speed) <= 0:
                r_lon = 0.5 + 0.01 * (vehicle.speed - cur_lane.min_speed)

            if self._agent_is_terminal(vehicle):
                r_lateral = r_lon = -100 if vehicle.goal is None else 0

        # Take fast lane
        if vehicle.lane_index[2] == 0: # right lane
            r_lane = 0.5
        elif vehicle.lane_index[2] == 1: # left lane
            r_lane = 1
            
        reward = r_lateral + r_lon + r_lane

        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        agent_info = []
        local_info = []

        obs, reward, done, info = super().step(action)
        info["agents_dones"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles
        )
        for v in self.controlled_vehicles:
            agent_info.append([v.position[0], v.position[1], v.speed])
            local_coords = v.lane.local_coordinates(v.position)
            local_info.append([local_coords[0], local_coords[1], v.speed])
            if np.isnan(v.position).any():
                print("Vehicle action: ", action)
                raise ValueError("Vehicle position is NaN")
        info["agents_info"] = agent_info
        info["hdv_info"] = local_info

        # hdv_info = []
        # for v in self.road.vehicles:
        #     if v not in self.controlled_vehicles:
        #         hdv_info.append([v.position[0], v.position[1], v.speed])
        # info["hdv_info"] = hdv_info

        for vehicle in self.controlled_vehicles:
            vehicle.local_reward = self._agent_reward(action, vehicle)
        # local reward
        info["regional_rewards"] = tuple(
            vehicle.local_reward for vehicle in self.controlled_vehicles
        )

        obs = np.asarray(obs).reshape((len(obs), -1))
        return obs, reward, done, info
    
    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(any(vehicle.crashed is False for vehicle in self.controlled_vehicles) )
    
    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (
            any(vehicle.crashed for vehicle in self.controlled_vehicles)
            or self.steps >= self.config["duration"] * self.config["policy_frequency"]
        )

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (
            vehicle.crashed
            or self.steps >= self.config["duration"] * self.config["policy_frequency"]
        )
    
    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], length= self.config["length"], min_speeds = self.config["min_speeds"], max_speeds = self.config["max_speeds"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])
        
    def _create_goal(self) -> None:
        """Create a goal region on the straight lane for the LC vehicle"""
        lane = self.road.network.get_lane(("0", "1", self.config["target_lane"]))
        self.goal = Landmark.make_on_lane(lane=lane, longitudinal=lane.length - Vehicle.LENGTH)
        self.goal.set_target_lane(self.config["target_lane"])
        self.road.objects.append(self.goal)

    def _create_vehicles(self) -> None:
        """Create a central vehicle and four other AVs surrounding a main vehicle in random positions."""

        road = self.road
        lane_count = self.config["lanes_count"]
        init_spawn_length = self.config["length"] / 3
        self.controlled_vehicles = []

        lc_spawn_pos = init_spawn_length/2
        # lc_spawn_pos = np.random.choice([-3, -2, -1, 0, 1, 2, 3]) * Vehicle.LENGTH + lc_spawn_pos
        

        # initial speed with noise and location noise
        initial_speed = list (
            np.random.rand(self.config["controlled_vehicles"]) * 2 + 25
        )  # range from [25, 27]


        # Add first vehicle to perform lane change
        target_lane_index = self.config["target_lane"]
        
        # Spwan in lane other than target lane
        # lc_vehicle_spwan_lane = (target_lane_index + np.random.choice([0,1])) % lane_count
        lc_vehicle_spwan_lane = (target_lane_index + 1) % lane_count
        # lc_vehicle_spwan_lane = target_lane_index

        lc_vehicle: ControlledBicycleVehicle = self.action_type.vehicle_class(
                road = road,
                position = road.network.get_lane(("0", "1", lc_vehicle_spwan_lane)).position(
                    lc_spawn_pos, 0
                ),
                speed = initial_speed.pop(0),
            )
        
        lc_vehicle.set_target_lane(target_lane_index)
        lc_vehicle.set_goal(self.goal)

        self.controlled_vehicles.append(lc_vehicle)
        road.vehicles.append(lc_vehicle)

        # print("LC Vehicle: Spawn lane: {}, Position: {}".format(lc_vehicle_spwan_lane, lc_vehicle.position))

        # Add autonomous vehicles to follow lane
        n_follow_vehicle = self.config["controlled_vehicles"] - 1
        spawn_points = np.random.rand(n_follow_vehicle)
        # CAVs in behind
        spawn_points[:2] = spawn_points[:2] * lc_spawn_pos - 2*Vehicle.LENGTH

        # CAVs front
        spawn_points[2:] = 2*Vehicle.LENGTH + lc_spawn_pos + spawn_points[2:] * (init_spawn_length - lc_spawn_pos)

        spawn_points = list(spawn_points)

        for idx in range(n_follow_vehicle):
            lane_id = idx % lane_count
            lane_follow_vehicle = self.action_type.vehicle_class(
                road = road,
                position = road.network.get_lane(("0", "1", lane_id)).position(
                    spawn_points.pop(0), 0
                ),
                speed = initial_speed.pop(0),
            )
            self.controlled_vehicles.append(lane_follow_vehicle)
            road.vehicles.append(lane_follow_vehicle)
            # print("Other Vehicles: Spawn lane: {}, Position: {}".format(lane_id, lane_follow_vehicle.position))

    def define_spaces(self) -> None:
        """
        Define spaces of agents and observations
        """
        super().define_spaces()
        # enable only first CAV to move laterally
        if len(self.action_type.agents_action_types) > 0:
            self.action_type.agents_action_types[0].lateral = True

    def is_vehicle_on_road(self, vehicle: Vehicle) -> bool:
        if vehicle.position[0] <= 0.0:
            return False
        # center of second lane is 0, so accomodate half of the lane width
        if vehicle.position[1] <= -0.5 * AbstractLane.DEFAULT_WIDTH:
            return False
        # center of second lane is AbstractLane.DEFAULT_WIDTH
        if vehicle.position[1] >= 1.5 * AbstractLane.DEFAULT_WIDTH:
            return False
        # if vehicle.position[0] > self.config["length"]: let the vehicle go out of the max length
        return True
register(
    id="lanechange-marl-v0",
    entry_point="highway_env.envs:LaneChnageMARL",
)


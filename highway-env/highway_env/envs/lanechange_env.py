import numpy as np
from gym.envs.registration import register
import math
from typing import Tuple

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle


class LaneChnageMARL(AbstractEnv):

    n_a = 2
    n_s = 25
    
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "MultiAgentAction",
                    "action_config": {
                        "type": "ContinuousAction",
                        "lateral": True,
                        "longitudinal": True,
                        "dynamical": True,
                    },
                },
                "observation": {
                    "type": "MultiAgentObservation",
                    "observation_config": {"type": "Kinematics"},
                },
                "lanes_count": 2,
                "controlled_vehicles": 5,
                "safety_guarantee": False,
                "action_masking": False,
                "target_lane": 0,
                "initial_lane_id": 1,
                "length": 300,
                "screen_width": 1200,
                "screen_height": 100,
                # "centering_position": [0, 4],
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
        self._create_vehicles()
        # self.action_is_safe = True
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        return sum(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        ) / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
        The first vehicle is rewarded for 
            - moving towards the middle of target lane,
        All the vehicles are rewarded for
            - moving forward,
            - high speed,
            - headway distance, 
            - avoiding collisions,
            - avoid going off road boundaries.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        last_pos = vehicle.position.copy()
        if len(vehicle.history) > 1:
            last_pos = vehicle.history.popleft()
        
        # reward for moving forward
        dx = vehicle.position[0] - last_pos[0]
        dx_s = utils.lmap(dx, [0, vehicle.LENGTH], [0, 1])
        
        # reward for moving towards the middle of target lane
        # Optimal reward 0
        target_lane_index = ("0", "1", self.config["target_lane"])
        if vehicle == self.controlled_vehicles[0]:
            target_lane = self.road.network.get_lane(target_lane_index)
            lane_coords = target_lane.local_coordinates(self.vehicle.position)
            dy = abs(vehicle.position[1] - lane_coords[1])
            dy_s = utils.lmap(dy, [0, 0.5* AbstractLane.DEFAULT_WIDTH], [0, 1])
        else:
            dy_s = 0

        # the optimal reward is 1
        speed_s = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )

        # compute headway cost
        headway_distance = self._compute_headway_distance(vehicle)
        headway_cost = (
            np.log(headway_distance / (self.config["HEADWAY_TIME"] * vehicle.speed))
            if vehicle.speed > 0
            else 0
        )

        # is on the road
        is_off_lane = 1 - self.is_vehicle_on_road(vehicle)

        # compute overall reward
        reward = (
            self.config["LATERAL_MOTION_REWARD"] * dy_s
            + self.config["LONGITUDINAL_MOTION_REWARD"] * dx_s
            + self.config["HIGH_SPEED_REWARD"] * speed_s
            + self.config["HEADWAY_COST"] * (headway_cost if headway_cost < 0 else 0)
            + self.config["COLLISION_COST"] * (-1 * (vehicle.crashed + is_off_lane))
        )
        return reward

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], length= self.config["length"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

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
        info["agents_rewards"] = tuple(
            vehicle.local_reward for vehicle in self.controlled_vehicles
        )

        obs = np.asarray(obs).reshape((len(obs), -1))
        return obs, reward, done, info
    
    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(any(vehicle.on_road is False for vehicle in self.controlled_vehicles) 
                     and any(self.is_vehicle_on_road(vehicle) is False for vehicle in self.controlled_vehicles))
    
    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (
            any(vehicle.crashed for vehicle in self.controlled_vehicles)
            or self.steps >= self.config["duration"] * self.config["policy_frequency"]
            or any(self.is_vehicle_on_road(vehicle) is False for vehicle in self.controlled_vehicles)
        )

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (
            vehicle.crashed
            or self.steps >= self.config["duration"] * self.config["policy_frequency"]
            or not self.is_vehicle_on_road(vehicle)
        )
    
    def _create_vehicles(self) -> None:
        """Create a central vehicle and four other AVs surrounding a main vehicle in random positions."""

        road = self.road
        lane_count = self.config["lanes_count"]
        init_spawn_length = self.config["length"] / 3
        lc_spawn_pos = init_spawn_length/2
        self.controlled_vehicles = []

        # initial speed with noise and location noise
        initial_speed = list (
            np.random.rand(self.config["controlled_vehicles"]) * 2 + 25
        )  # range from [25, 27]


        # Add first vehicle to perform lane change

        lc_vehicle_spwan = lc_spawn_pos
        target_lane_index = self.config["target_lane"]
        # Spwan in lane other than target lane
        lc_vehicle_spwan_lane = (target_lane_index + 1) % lane_count
        lc_vehicle = self.action_type.vehicle_class(
                road,
                road.network.get_lane(("0", "1", lc_vehicle_spwan_lane)).position(
                    lc_vehicle_spwan, 0
                ),
                speed=initial_speed.pop(0),
            )
        self.controlled_vehicles.append(lc_vehicle)
        road.vehicles.append(lc_vehicle)

        # Add autonomous vehicles to follow lane
        n_follow_vehicle = self.config["controlled_vehicles"] - 1
        spawn_points = np.random.rand(n_follow_vehicle)
        spawn_points[:2] = spawn_points[:2] * lc_spawn_pos - Vehicle.LENGTH
        spawn_points[2:] = Vehicle.LENGTH + lc_spawn_pos + spawn_points[2:] * (init_spawn_length - lc_spawn_pos)
        # randomize the order of adding a vehicle
        spawn_points = np.random.choice(spawn_points, n_follow_vehicle, replace=False)
        spawn_points = list(spawn_points)

        for idx in range(n_follow_vehicle):
            lane_id = idx % lane_count
            lane_follow_vehicle = self.action_type.vehicle_class(
                road,
                road.network.get_lane(("0", "1", lane_id)).position(
                    spawn_points.pop(0), 0
                ),
                speed=initial_speed.pop(0),
            )
            self.controlled_vehicles.append(lane_follow_vehicle)
            road.vehicles.append(lane_follow_vehicle)
        
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


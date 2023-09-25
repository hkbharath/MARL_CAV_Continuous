import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                          [self.config["collision_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)
    


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False

class HighwayEnvContinuousMARL(HighwayEnvFast):

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
                "controlled_vehicles": 1,
                "safety_guarantee": False,
                "action_masking": False,
            }
        )
        return config
    
    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        return sum(
            self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles
        ) / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        # the optimal reward is 0
        scaled_speed = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        # compute cost for staying on the merging lane
        if vehicle.lane_index == ("b", "c", 1):
            Merging_lane_cost = -np.exp(
                -((vehicle.position[0] - sum(self.ends[:3])) ** 2) / (10 * self.ends[2])
            )
        else:
            Merging_lane_cost = 0

        # compute headway cost
        headway_distance = self._compute_headway_distance(vehicle)
        Headway_cost = (
            np.log(headway_distance / (self.config["HEADWAY_TIME"] * vehicle.speed))
            if vehicle.speed > 0
            else 0
        )
        # compute overall reward
        reward = (
            self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed)
            + (self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1))
            + self.config["MERGING_LANE_COST"] * Merging_lane_cost
            + self.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0)
            + self.config["COLLISION_REWARD"] * (-1 * (not self.road.network.get_lane(vehicle.lane_index).on_lane(vehicle.position)))
        )
        return reward

    def _regional_reward(self):
        for vehicle in self.controlled_vehicles:
            neighbor_vehicle = []

            # vehicle is on the main road
            if (
                vehicle.lane_index == ("a", "b", 0)
                or vehicle.lane_index == ("b", "c", 0)
                or vehicle.lane_index == ("c", "d", 0)
            ):
                v_fl, v_rl = self.road.surrounding_vehicles(vehicle)
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fr, v_rr = self.road.surrounding_vehicles(
                        vehicle, self.road.network.side_lanes(vehicle.lane_index)[0]
                    )
                # assume we can observe the ramp on this road
                elif (
                    vehicle.lane_index == ("a", "b", 0)
                    and vehicle.position[0] > self.ends[0]
                ):
                    v_fr, v_rr = self.road.surrounding_vehicles(vehicle, ("k", "b", 0))
                else:
                    v_fr, v_rr = None, None
            else:
                # vehicle is on the ramp
                v_fr, v_rr = self.road.surrounding_vehicles(vehicle)
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fl, v_rl = self.road.surrounding_vehicles(
                        vehicle, self.road.network.side_lanes(vehicle.lane_index)[0]
                    )
                # assume we can observe the straight road on the ramp
                elif vehicle.lane_index == ("k", "b", 0):
                    v_fl, v_rl = self.road.surrounding_vehicles(vehicle, ("a", "b", 0))
                else:
                    v_fl, v_rl = None, None
            for v in [v_fl, v_fr, vehicle, v_rl, v_rr]:
                if v is not None and (
                    type(v) is MDPVehicle or isinstance(v, BicycleVehicle)
                ):
                    neighbor_vehicle.append(v)
            vehicle.regional_reward = 0
            if len(neighbor_vehicle) > 0:
                regional_reward = sum(v.local_reward for v in neighbor_vehicle)
                vehicle.regional_reward = regional_reward / sum(
                    1 for _ in filter(None.__ne__, neighbor_vehicle)
                )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        agent_info = []
        obs, reward, done, info = super().step(action)
        info["agents_dones"] = tuple(
            self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles
        )
        for v in self.controlled_vehicles:
            agent_info.append([v.position[0], v.position[1], v.speed])
            if np.isnan(v.position).any():
                raise ValueError("Vehicle position is NaN")
        info["agents_info"] = agent_info

        hdv_info = []
        for v in self.road.vehicles:
            if v not in self.controlled_vehicles:
                hdv_info.append([v.position[0], v.position[1], v.speed])
        info["hdv_info"] = hdv_info

        for vehicle in self.controlled_vehicles:
            vehicle.local_reward = self._agent_reward(action, vehicle)
        # local reward
        info["agents_rewards"] = tuple(
            vehicle.local_reward for vehicle in self.controlled_vehicles
        )
        # regional reward
        self._regional_reward()
        info["regional_rewards"] = tuple(
            vehicle.regional_reward for vehicle in self.controlled_vehicles
        )

        obs = np.asarray(obs).reshape((len(obs), -1))
        return obs, reward, done, info
    
    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(any(vehicle.on_road is False for vehicle in self.controlled_vehicles))
    
    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (
            any(vehicle.crashed for vehicle in self.controlled_vehicles)
            or self.steps >= self.config["duration"] * self.config["policy_frequency"]
            or any(vehicle.on_road is False for vehicle in self.controlled_vehicles)
        )

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (
            vehicle.crashed
            or self.steps >= self.config["duration"] * self.config["policy_frequency"]
            or not vehicle.on_road
        )
    
    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        
        # Replace this with planned structure.
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

register(
    id='highway-v0',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-fast-v0',
    entry_point='highway_env.envs:HighwayEnvFast',
)

register(
    id="highway-continuous-marl-v0",
    entry_point="highway_env.envs:HighwayEnvContinuousMARL",
)

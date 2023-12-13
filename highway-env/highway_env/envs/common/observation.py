from typing import List, Dict, TYPE_CHECKING, Optional, Union
from gym import spaces
import gym
gym.logger.set_level(40)
import numpy as np
import pandas as pd

from highway_env import utils
from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.dynamics import ControlledBicycleVehicle
from highway_env.road.objects import Landmark

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


class ObservationType(object):
    def __init__(self, env: 'AbstractEnv', **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()

    @property
    def observer_vehicle(self):
        """
        The vehicle observing the scene.

        If not set, the first controlled vehicle is used by default.
        """
        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle


class GrayscaleObservation(ObservationType):

    """
    An observation class that collects directly what the simulator renders

    Also stacks the collected frames as in the nature DQN.
    Specific keys are expected in the configuration dictionary passed.

    Example of observation dictionary in the environment config:
        observation": {
            "type": "GrayscaleObservation",
            "weights": [0.2989, 0.5870, 0.1140],  #weights for RGB conversion,
            "stack_size": 4,
            "observation_shape": (84, 84)
        }

    Also, the screen_height and screen_width of the environment should match the
    expected observation_shape.
    """

    def __init__(self, env: 'AbstractEnv', config: dict) -> None:
        super().__init__(env)
        self.config = config
        self.observation_shape = config["observation_shape"]
        self.shape = self.observation_shape + (config["stack_size"], )
        self.state = np.zeros(self.shape)

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(shape=self.shape,
                              low=0, high=1,
                              dtype=np.float32)
        except AttributeError:
            return spaces.Space()

    def observe(self) -> np.ndarray:
        new_obs = self._record_to_grayscale()
        new_obs = np.reshape(new_obs, self.observation_shape)
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[:, :, -1] = new_obs
        return self.state

    def _record_to_grayscale(self) -> np.ndarray:
        #TODO: center rendering on the observer vehicle
        raw_rgb = self.env.render('rgb_array')
        return np.dot(raw_rgb[..., :3], self.config['weights'])


class TimeToCollisionObservation(ObservationType):
    def __init__(self, env: 'AbstractEnv', horizon: int = 10, **kwargs: dict) -> None:
        super().__init__(env)
        self.horizon = horizon

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(shape=self.observe().shape, low=0, high=1, dtype=np.float32)
        except AttributeError:
            return spaces.Space()

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros((3, 3, int(self.horizon * self.env.config["policy_frequency"])))
        grid = compute_ttc_grid(self.env, vehicle=self.observer_vehicle,
                                time_quantization=1/self.env.config["policy_frequency"], horizon=self.horizon)
        padding = np.ones(np.shape(grid))
        padded_grid = np.concatenate([padding, grid, padding], axis=1)
        obs_lanes = 3
        l0 = grid.shape[1] + self.observer_vehicle.lane_index[2] - obs_lanes // 2
        lf = grid.shape[1] + self.observer_vehicle.lane_index[2] + obs_lanes // 2
        clamped_grid = padded_grid[:, l0:lf+1, :]
        repeats = np.ones(clamped_grid.shape[0])
        repeats[np.array([0, -1])] += clamped_grid.shape[0]
        padded_grid = np.repeat(clamped_grid, repeats.astype(int), axis=0)
        obs_speeds = 3
        v0 = grid.shape[0] + self.observer_vehicle.speed_index - obs_speeds // 2
        vf = grid.shape[0] + self.observer_vehicle.speed_index + obs_speeds // 2
        clamped_grid = padded_grid[v0:vf + 1, :, :]
        return clamped_grid


class KinematicObservation(ObservationType):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = False,
                 see_behind: bool = True,
                 observe_intentions: bool = False,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

    def space(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-1, high=1, dtype=np.float32)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            # side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            # self.features_range = {
            #     "x": [-5.0 * MDPVehicle.SPEED_MAX, 5.0 * MDPVehicle.SPEED_MAX],
            #     "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
            #     "vx": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX],
            #     "vy": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX]
            # }
            self.features_range = {
                "x": [-5.0 * MDPVehicle.SPEED_MAX, 5.0 * MDPVehicle.SPEED_MAX],
                "y": [-12, 12],
                "vx": [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX],
                "vy": [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        # Add nearby traffic
        # sort = self.order == "sorted"
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind)
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = df.append(pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs


class OccupancyGridObservation(ObservationType):

    """Observe an occupancy grid of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'vx', 'vy']
    GRID_SIZE: List[List[float]] = [[-5.5*5, 5.5*5], [-5.5*5, 5.5*5]]
    GRID_STEP: List[int] = [5, 5]

    def __init__(self,
                 env: 'AbstractEnv',
                 features: Optional[List[str]] = None,
                 grid_size: Optional[List[List[float]]] = None,
                 grid_step: Optional[List[int]] = None,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        """
        super().__init__(env)
        self.features = features if features is not None else self.FEATURES
        self.grid_size = np.array(grid_size) if grid_size is not None else np.array(self.GRID_SIZE)
        self.grid_step = np.array(grid_step) if grid_step is not None else np.array(self.GRID_STEP)
        grid_shape = np.asarray(np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / grid_step), dtype=np.int)
        self.grid = np.zeros((len(self.features), *grid_shape))
        self.features_range = features_range
        self.absolute = absolute

    def space(self) -> spaces.Space:
        return spaces.Box(shape=self.grid.shape, low=-1, high=1, dtype=np.float32)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            self.features_range = {
                "vx": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX],
                "vy": [-2*MDPVehicle.SPEED_MAX, 2*MDPVehicle.SPEED_MAX]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        if self.absolute:
            raise NotImplementedError()
        else:
            # Add nearby traffic
            self.grid.fill(0)
            df = pd.DataFrame.from_records(
                [v.to_dict(self.observer_vehicle) for v in self.env.road.vehicles])
            # Normalize
            df = self.normalize(df)
            # Fill-in features
            for layer, feature in enumerate(self.features):
                for _, vehicle in df.iterrows():
                    x, y = vehicle["x"], vehicle["y"]
                    # Recover unnormalized coordinates for cell index
                    if "x" in self.features_range:
                        x = utils.lmap(x, [-1, 1], [self.features_range["x"][0], self.features_range["x"][1]])
                    if "y" in self.features_range:
                        y = utils.lmap(y, [-1, 1], [self.features_range["y"][0], self.features_range["y"][1]])
                    cell = (int((x - self.grid_size[0, 0]) / self.grid_step[0]),
                            int((y - self.grid_size[1, 0]) / self.grid_step[1]))
                    if 0 <= cell[1] < self.grid.shape[-2] and 0 <= cell[0] < self.grid.shape[-1]:
                        self.grid[layer, cell[1], cell[0]] = vehicle[feature]
            # Clip
            obs = np.clip(self.grid, -1, 1)
            return obs


class KinematicsGoalObservation(KinematicObservation):
    def __init__(self, env: 'AbstractEnv', scales: List[float], **kwargs: dict) -> None:
        self.scales = np.array(scales)
        super().__init__(env, **kwargs)

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
                observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
            ))
        except AttributeError:
            return spaces.Space()

    def observe(self) -> Dict[str, np.ndarray]:
        if not self.observer_vehicle:
            return {
            "observation": np.zeros((len(self.features),)),
            "achieved_goal": np.zeros((len(self.features),)),
            "desired_goal": np.zeros((len(self.features),))
        }

        obs = np.ravel(pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features])
        goal = np.ravel(pd.DataFrame.from_records([self.env.goal.to_dict()])[self.features])
        obs = {
            "observation": obs / self.scales,
            "achieved_goal": obs / self.scales,
            "desired_goal": goal / self.scales
        }
        return obs


class AttributesObservation(ObservationType):
    def __init__(self, env: 'AbstractEnv', attributes: List[str], **kwargs: dict) -> None:
        self.env = env
        self.attributes = attributes

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict({
                attribute: spaces.Box(-np.inf, np.inf, shape=obs[attribute].shape, dtype=np.float32)
                for attribute in self.attributes
            })
        except AttributeError:
            return spaces.Space()

    def observe(self) -> Dict[str, np.ndarray]:
        return {
            attribute: getattr(self.env, attribute) for attribute in self.attributes
        }


class MultiAgentObservation(ObservationType):
    def __init__(self,
                 env: 'AbstractEnv',
                 observation_config: dict,
                 **kwargs) -> None:
        super().__init__(env)
        self.observation_config = observation_config
        self.agents_observation_types = []
        for vehicle in self.env.controlled_vehicles:
            obs_type = observation_factory(self.env, self.observation_config)
            obs_type.observer_vehicle = vehicle
            self.agents_observation_types.append(obs_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple([obs_type.space() for obs_type in self.agents_observation_types])

    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.agents_observation_types)


def observation_factory(env: 'AbstractEnv', config: dict) -> ObservationType:
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return GrayscaleObservation(env, config)
    elif config["type"] == "AttributesObservation":
        return AttributesObservation(env, **config)
    elif config["type"] == "MultiAgentObservation":
        return MultiAgentObservation(env, **config)
    elif config["type"] == "KinameticObsExt":
        return KinameticObsExt(env, **config)
    else:
        raise ValueError("Unknown observation type")

# TODO: evaluate and collect observations
class KinameticObsExt(ObservationType):

    """Observe the kinematics of nearby vehicles and distance from road edges"""

    #['front wheel angle', 'speed', 'acceleration']
    # * - host vehicle
    # lf - left front vehicle  
    # lb - left back vehicle
    # rf - right front vehicle
    # rb - right back vehicle
    CAR_FEATURES: List[str] = ['del', 'v', 'a', 'del_lf', 'v_lf', 'a_lf', 'del_lb', 'v_lb', 'a_lb','del_rf', 'v_rf', 'a_rf', 'del_rb', 'v_rb', 'a_rb']
    
    #['angle', 'left distance', 'middle distance', 'right distance', 
    # 'current_lane', 'left_lane_v_upper', 'left_lane_v_lower', 'right_lane_v_upper', 'right_lane_v_lower']
    ROAD_FEATURES: List[str] = ['phi', 'dl', 'dm', 'dr',
                                'curr_lane', 'left_lane_vu', 'left_lane_vl', 'right_lane_vu', 'right_lane_vl']

    DESTINATION_FEATURES: List[str]  = ['dest_lane', 'dest_dist'] 

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 4,
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = True,
                 observe_intentions: bool = False,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.vehicles_count = vehicles_count
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions
        self.features = features if features is not None else self.CAR_FEATURES + self.ROAD_FEATURES + self.DESTINATION_FEATURES

    def space(self) -> spaces.Space:
        return spaces.Box(shape=(1, len(self.features)), low=-1, high=1, dtype=np.float32)

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)
        
        obs_dict = {el: 0 for el in self.features}

        host_vehicle: ControlledBicycleVehicle = self.observer_vehicle

        # Add ego-vehicle
        obs_dict['del'] = host_vehicle.heading
        obs_dict['v'] = host_vehicle.speed
        obs_dict['a'] = host_vehicle.action['acceleration']

        # Add nearby traffic
        # sort = self.order == "sorted"
        close_vehicles = self.env.road.close_vehicles_to(host_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind)
        # set default distance if vehicle does nto exist
        obs_dict['del_rf'] = 2 * host_vehicle.LENGTH
        obs_dict['del_lf'] = 2 * host_vehicle.LENGTH
        obs_dict['del_rb'] = 2 * host_vehicle.LENGTH
        obs_dict['del_lb'] = 2 * host_vehicle.LENGTH

        if close_vehicles:
            for v in close_vehicles[-self.vehicles_count + 1:]:
                vf = v.to_dict(host_vehicle, observe_intentions=self.observe_intentions)
                if vf['x'] > 0:
                    if v.lane_index[2] == 0:
                        obs_dict['del_rf'] = vf['x']
                        obs_dict['v_rf'] = v.speed
                        obs_dict['a_rf'] = v.action['acceleration']
                    elif v.lane_index[2] == 1:
                        obs_dict['del_lf'] = vf['x']
                        obs_dict['v_lf'] = v.speed
                        obs_dict['a_lf'] = v.action['acceleration']
                else:
                    if v.lane_index[2] == 0:
                        obs_dict['del_rb'] = vf['x']
                        obs_dict['v_rb'] = v.speed
                        obs_dict['a_rb'] = v.action['acceleration']
                    elif v.lane_index[2] == 1:
                        obs_dict['del_lb'] = vf['x']
                        obs_dict['v_lb'] = v.speed
                        obs_dict['a_lb'] = v.action['acceleration']

        # Add road features
        obs_dict['phi'] = host_vehicle.lane.heading_at(host_vehicle.position[0])
        # Fix these
        obs_dict['dl'] = host_vehicle.dist_to_left()
        obs_dict['dm'] = host_vehicle.dist_to_mid()
        obs_dict['dr'] = host_vehicle.dist_to_right()

        obs_dict['curr_lane'] = host_vehicle.lane_index[2]

        lanes_list = self.env.road.network.lanes_list()
        obs_dict['left_lane_vu'] = lanes_list[1].speed_limit
        obs_dict['left_lane_vl'] = lanes_list[1].min_speed
        obs_dict['right_lane_vu'] = lanes_list[0].speed_limit
        obs_dict['right_lane_vl'] = lanes_list[0].min_speed

        for ob in self.env.road.objects:
            if isinstance(ob, Landmark):
                dest:Landmark = ob
                # print("Goal lane number: {}".format(type(dest.lane_num)))
                obs_dict['dest_lane'] = dest.lane_num
                obs_dict['dest_dist'] = dest.position[0] - host_vehicle.position[0]

        # print("Observations:")
        # print(obs_dict)

        obs = np.array([[val if val else 0 for val in obs_dict.values()]])

        return obs

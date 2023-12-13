from typing import Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from highway_env.road.road import Road
from highway_env.road.lane import AbstractLane
from highway_env.types import Vector
from highway_env.vehicle.kinematics import Vehicle
from highway_env import utils
from highway_env.road.objects import Landmark


class BicycleVehicle(Vehicle):
    """
    A dynamical bicycle model, with tire friction and slipping.
    
    See Chapter 2 of Lateral Vehicle Dynamics. Vehicle Dynamics and Control. Rajamani, R. (2011)
    """
    MASS: float = 1  # [kg]
    LENGTH_A: float = Vehicle.LENGTH / 2  # [m]
    LENGTH_B: float = Vehicle.LENGTH / 2  # [m]
    INERTIA_Z: float = 1/12 * MASS * (Vehicle.LENGTH ** 2 + 3 * Vehicle.WIDTH ** 2)  # [kg.m2]
    FRICTION_FRONT: float = 15.0 * MASS  # [N]
    FRICTION_REAR: float = 15.0 * MASS  # [N]

    MAX_ANGULAR_SPEED: float = 2 * np.pi  # [rad/s] 2* np.pi before
    MAX_SPEED: float = 40  # [m/s] 15 before
    MAX_STEERING_ANGLE: float = np.pi / 2  # [rad] np.pi/2 before
    MAX_ACCELERATION: float = 3  # [m/s2] 

    def __init__(self, road: Road, position: Vector, heading: float = 0, speed: float = 0) -> None:
        super().__init__(road, position, heading, speed)
        self.lateral_speed = 0
        self.yaw_rate = 0
        self.theta = None
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()

    @property
    def state(self) -> np.ndarray:
        return np.array([[self.position[0]],
                         [self.position[1]],
                         [self.heading],
                         [self.speed],
                         [self.lateral_speed],
                         [self.yaw_rate]])

    @property
    def derivative(self) -> np.ndarray:
        """
        See Chapter 2 of Lateral Vehicle Dynamics. Vehicle Dynamics and Control. Rajamani, R. (2011)

        :return: the state derivative
        """
        delta_f = self.action["steering"]
        delta_r = 0
        theta_vf = np.arctan2(self.lateral_speed + self.LENGTH_A * self.yaw_rate, self.speed)  # (2.27)
        theta_vr = np.arctan2(self.lateral_speed - self.LENGTH_B * self.yaw_rate, self.speed)  # (2.28)
        f_yf = 2*self.FRICTION_FRONT * (delta_f - theta_vf)  # (2.25)
        f_yr = 2*self.FRICTION_REAR * (delta_r - theta_vr)  # (2.26)
        if abs(self.speed) < 1:  # Low speed dynamics: damping of lateral speed and yaw rate
            f_yf = - self.MASS * self.lateral_speed - self.INERTIA_Z/self.LENGTH_A * self.yaw_rate
            f_yr = - self.MASS * self.lateral_speed + self.INERTIA_Z/self.LENGTH_A * self.yaw_rate
        d_lateral_speed = 1/self.MASS * (f_yf + f_yr) - self.yaw_rate * self.speed  # (2.21)
        d_yaw_rate = 1/self.INERTIA_Z * (self.LENGTH_A * f_yf - self.LENGTH_B * f_yr)  # (2.22)
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array(((c, -s), (s, c)))
        speed = R @ np.array([self.speed, self.lateral_speed])
        return np.array([[speed[0]],
                         [speed[1]],
                         [self.yaw_rate],
                         [self.action['acceleration']],
                         [d_lateral_speed],
                         [d_yaw_rate]])

    @property
    def derivative_linear(self) -> np.ndarray:
        """
        Linearized lateral dynamics.
            
        This model is based on the following assumptions:
        - the vehicle is moving with a constant longitudinal speed
        - the steering input to front tires and the corresponding slip angles are small
        
        See https://pdfs.semanticscholar.org/bb9c/d2892e9327ec1ee647c30c320f2089b290c1.pdf, Chapter 3.
        """
        x = np.array([[self.lateral_speed], [self.yaw_rate]])
        u = np.array([[self.action['steering']]])
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()
        dx = self.A_lat @ x + self.B_lat @ u
        c, s = np.cos(self.heading), np.sin(self.heading)
        R = np.array(((c, -s), (s, c)))
        speed = R @ np.array([self.speed, self.lateral_speed])
        return np.array([[speed[0]], [speed[1]], [self.yaw_rate], [self.action['acceleration']], dx[0], dx[1]])

    def step(self, dt: float) -> None:
        self.clip_actions()
        derivative = self.derivative
        
        new_pos = self.position + derivative[0:2, 0] * dt
        # if not self.lane.on_lane(new_pos):
        #     # print("going off road...")
        #     # self.action = {"acceleration": 0, "steering": 0}
        #     # derivative = self.derivative
        #    return
        
        self.position += derivative[0:2, 0] * dt
        self.heading += self.yaw_rate * dt
        self.speed += self.action['acceleration'] * dt
        self.lateral_speed += derivative[4, 0] * dt
        self.yaw_rate += derivative[5, 0] * dt

        self.on_state_update()

    def clip_actions(self) -> None:
        super().clip_actions()
        # Required because of the linearisation
        self.action["steering"] = np.clip(self.action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        self.yaw_rate = np.clip(self.yaw_rate, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)

    def lateral_lpv_structure(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        State: [lateral speed v, yaw rate r]

        :return: lateral dynamics A0, phi, B such that dx = (A0 + theta^T phi)x + B u
        """
        B = np.array([
            [2*self.FRICTION_FRONT / self.MASS],
            [self.FRICTION_FRONT * self.LENGTH_A / self.INERTIA_Z]
        ])

        speed_body_x = self.speed
        A0 = np.array([
            [0, -speed_body_x],
            [0, 0]
        ])

        if abs(speed_body_x) < 1:
            return A0, np.zeros((2, 2, 2)), B*0

        phi = np.array([
            [
                [-2 / (self.MASS*speed_body_x), -2*self.LENGTH_A / (self.MASS*speed_body_x)],
                [-2*self.LENGTH_A / (self.INERTIA_Z*speed_body_x), -2*self.LENGTH_A**2 / (self.INERTIA_Z*speed_body_x)]
            ], [
                [-2 / (self.MASS*speed_body_x), 2*self.LENGTH_B / (self.MASS*speed_body_x)],
                [2*self.LENGTH_B / (self.INERTIA_Z*speed_body_x), -2*self.LENGTH_B**2 / (self.INERTIA_Z*speed_body_x)]
            ],
        ])
        return A0, phi, B

    def lateral_lpv_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        State: [lateral speed v, yaw rate r]

        :return: lateral dynamics A, B
        """
        A0, phi, B = self.lateral_lpv_structure()
        self.theta = np.array([self.FRICTION_FRONT, self.FRICTION_REAR])
        A = A0 + np.tensordot(self.theta, phi, axes=[0, 0])
        return A, B

    def full_lateral_lpv_structure(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        State: [position y, yaw psi, lateral speed v, yaw rate r]

        The system is linearized around psi = 0

        :return: lateral dynamics A, phi, B
        """
        A_lat, phi_lat, B_lat = self.lateral_lpv_structure()

        speed_body_x = self.speed
        A_top = np.array([
            [0, speed_body_x, 1, 0],
            [0, 0, 0, 1]
        ])
        A0 = np.concatenate((A_top, np.concatenate((np.zeros((2, 2)), A_lat), axis=1)))
        phi = np.array([np.concatenate((np.zeros((2, 4)), np.concatenate((np.zeros((2, 2)), phi_i), axis=1)))
                        for phi_i in phi_lat])
        B = np.concatenate((np.zeros((2, 1)), B_lat))
        return A0, phi, B

    def full_lateral_lpv_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        State: [position y, yaw psi, lateral speed v, yaw rate r]

        The system is linearized around psi = 0

        :return: lateral dynamics A, B
        """
        A0, phi, B = self.full_lateral_lpv_structure()
        self.theta = [self.FRICTION_FRONT, self.FRICTION_REAR]
        A = A0 + np.tensordot(self.theta, phi, axes=[0, 0])
        return A, B

class ControlledBicycleVehicle(BicycleVehicle):
    
    TAU_DS = 0.6  # [s]
    PURSUIT_TAU = 0.5*TAU_DS  # [s]

    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1/3 * KP_HEADING  # [1/s]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.target_lane = self.lane
        self.goal = None
    
    def set_target_lane(self, lane_index: int) -> None:
        if self.road:
            _from, _to, id = self.lane_index
            target_lane_index = (_from, _to, lane_index)
            self.target_lane = self.road.network.get_lane(target_lane_index)

    def set_goal(self, goal: Landmark) -> None:
        self.goal = goal

    def dist_to_right(self) -> float:
        # assuming 2 lanes
        l_width = self.road.network.lanes_list()[0].width_at(0)
        if self.lane_index[2] == 0:
            return abs(self.lane.local_coordinates(self.position)[1] - l_width/2)
        elif self.lane_index[2] == 1:
            return abs(self.lane.local_coordinates(self.position)[1] - 3*l_width/2)
        
        raise ValueError("dist_to_right() called on a vehicle not on a valid 2 lane road")

    def dist_to_left(self) -> float:
        # assuming 2 lanes
        l_width = self.road.network.lanes_list()[0].width_at(0)
        if self.lane_index[2] == 1:
            return abs(self.lane.local_coordinates(self.position)[1] + l_width/2)
        elif self.lane_index[2] == 0:
            return abs(self.lane.local_coordinates(self.position)[1] + 3*l_width/2)
        
        raise ValueError("dist_to_right() called on a vehicle not on a valid 2 lane road")
    
    def dist_to_mid(self) -> float:
        # assuming 2 lanes
        l_width = self.road.network.lanes_list()[0].width_at(0)
        if self.lane_index[2] == 0:
            return abs(self.lane.local_coordinates(self.position)[1] + l_width/2)
        elif self.lane_index[2] == 1:
            return abs(self.lane.local_coordinates(self.position)[1] - l_width/2)
        
        raise ValueError("dist_to_right() called on a vehicle not on a valid 2 lane road")

    # def steering_control(self) -> float:
    #     """
    #     Steer the vehicle to follow the center of an given lane.

    #     1. Lateral position is controlled by a proportional controller yielding a lateral speed command
    #     2. Lateral speed command is converted to a heading reference
    #     3. Heading is controlled by a proportional controller yielding a heading rate command
    #     4. Heading rate command is converted to a steering angle

    #     :param target_lane_index: index of the lane to follow
    #     :return: a steering wheel angle command [rad]
    #     """
    #     target_lane = self.target_lane
    #     lane_coords = target_lane.local_coordinates(self.position)
    #     lane_next_coords = lane_coords[0] + self.speed * self.PURSUIT_TAU
    #     lane_future_heading = target_lane.heading_at(lane_next_coords)

    #     # Lateral position control
    #     lateral_speed_command = - self.KP_LATERAL * lane_coords[1]
    #     # Lateral speed to heading
    #     heading_command = np.arcsin(np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1))
    #     heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)
    #     # Heading control
    #     heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)
    #     # Heading rate to steering angle
    #     steering_angle = np.arcsin(np.clip(self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
    #                                        -1, 1))
    #     steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
    #     return float(steering_angle)
    
    # def act(self, action: Union[dict, str] = None) -> None:
    #     if action:
    #         if self.lane is self.target_lane:
    #             action["steering"] = -1*action["steering"]
    #         print("steering %f"%action["steering"])
    #     super().act(action)
    #     # self.action["steering"] = self.steering_control()


def simulate(dt: float = 0.1) -> None:
    import control
    time = np.arange(0, 20, dt)
    vehicle = BicycleVehicle(road=None, position=[0, 5], speed=8.3)
    xx, uu = [], []
    from highway_env.interval import LPV
    A, B = vehicle.full_lateral_lpv_dynamics()
    K = -np.asarray(control.place(A, B, -np.arange(1, 5)))
    lpv = LPV(x0=vehicle.state[[1, 2, 4, 5]].squeeze(), a0=A, da=[np.zeros(A.shape)], b=B,
              d=[[0], [0], [0], [1]], omega_i=[[0], [0]], u=None, k=K, center=None, x_i=None)

    for t in time:
        # Act
        u = K @ vehicle.state[[1, 2, 4, 5]]
        omega = 2*np.pi/20
        u_p = 0*np.array([[-20*omega*np.sin(omega*t) * dt]])
        u += u_p
        # Record
        xx.append(np.array([vehicle.position[0], vehicle.position[1], vehicle.heading])[:, np.newaxis])
        uu.append(u)
        # Interval
        lpv.set_control(u, state=vehicle.state[[1, 2, 4, 5]])
        lpv.step(dt)
        # x_i_t = lpv.change_coordinates(lpv.x_i_t, back=True, interval=True)
        # Step
        vehicle.act({"acceleration": 0, "steering": u})
        vehicle.step(dt)

    xx, uu = np.array(xx), np.array(uu)
    plot(time, xx, uu)


def plot(time: np.ndarray, xx: np.ndarray, uu: np.ndarray) -> None:
    pos_x, pos_y = xx[:, 0, 0], xx[:, 1, 0]
    psi_x, psi_y = np.cos(xx[:, 2, 0]), np.sin(xx[:, 2, 0])
    dir_x, dir_y = np.cos(xx[:, 2, 0] + uu[:, 0, 0]), np.sin(xx[:, 2, 0] + uu[:, 0, 0])
    _, ax = plt.subplots(1, 1)
    ax.plot(pos_x, pos_y, linewidth=0.5)
    dir_scale = 1/5
    ax.quiver(pos_x[::20]-0.5/dir_scale*psi_x[::20],
              pos_y[::20]-0.5/dir_scale*psi_y[::20],
              psi_x[::20], psi_y[::20],
              angles='xy', scale_units='xy', scale=dir_scale, width=0.005, headwidth=1)
    ax.quiver(pos_x[::20]+0.5/dir_scale*psi_x[::20], pos_y[::20]+0.5/dir_scale*psi_y[::20], dir_x[::20], dir_y[::20],
              angles='xy', scale_units='xy', scale=0.25, width=0.005, color='r')
    ax.axis("equal")
    ax.grid()
    plt.show()
    plt.close()


def main() -> None:
    simulate()


if __name__ == '__main__':
    main()

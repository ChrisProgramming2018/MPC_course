import numpy as np
import gym
import gym.spaces as spaces
from vehicle_planner import VehiclePlannerOptions, VehiclePlanner, NlpOptions
from dataclasses import dataclass
import warnings
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation, FuncAnimation
from animation import animate


@dataclass
class EnvironmentOptions:
    max_angle: float = np.pi * 5 / 11  # something, that is close to 90 degrees or pi/2 (but restricted from being 90
    initial_velocity_ego: float = 0  # Initial velocity of the ego vehicle
    initial_velocity_opposing: float = 0  # Initial velocity of the opposing vehicle
    initial_distance_long: float = -15
    initial_distance_lat: float = 0.5
    initial_pos_lat_ego: float = -1.5
    initial_pos_lon_ego: float = 20
    maximum_considered_vdiff: float = 10
    maximum_velocity_ego: float = 2
    maximum_velocity_opp: float = 4
    opp_cost_angle_s: float = np.pi / 4
    initial_ego_cost_angle_s: float = np.pi / 4
    initial_ego_cost_angle_n: float = 0
    index_of_simulated_steps: int = 7  # The mpc trajecectory is controlled until this index, before the rl agent takes
    max_iterations: int = 25
    sim_speed: float = 100
    # the next action


class RaceCarEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RaceCarEnv, self).__init__()  # Define action and observation space

        # Load options, change from standard variables and make sure the are defined well
        self.nlp_options = NlpOptions()
        self.ego_planner_options = VehiclePlannerOptions()
        self.opp_planner_options = VehiclePlannerOptions()
        self.env_options = EnvironmentOptions()
        assert (self.nlp_options.n_nodes >= self.env_options.index_of_simulated_steps)

        self.full_ego_trajectory = np.array([])  # size states/indexes/iterates
        self.full_opp_trajectory = np.array([])  # size states/indexes/iterates

        self.ego_planner_options.maximum_velocity_opposing = self.env_options.maximum_velocity_opp
        self.ego_planner_options.maximum_velocity_ego = self.env_options.maximum_velocity_ego

        self.opp_planner_options.maximum_velocity_opposing = self.env_options.maximum_velocity_ego
        self.opp_planner_options.maximum_velocity_ego = self.env_options.maximum_velocity_opp

        self.ego_planner = VehiclePlanner(self.ego_planner_options, self.nlp_options)
        self.opp_planner = VehiclePlanner(self.opp_planner_options, self.nlp_options)

        ego_state = np.array([self.env_options.initial_pos_lon_ego,
                              self.env_options.initial_velocity_ego,
                              self.env_options.initial_pos_lat_ego,
                              0])
        opp_state = np.array([self.env_options.initial_pos_lon_ego + self.env_options.initial_distance_long,
                              self.env_options.initial_velocity_opposing,
                              self.env_options.initial_pos_lat_ego + self.env_options.initial_distance_lat,
                              0])

        self.ego_planner.set_states(states_ego=ego_state, states_opp=opp_state)
        self.opp_planner.set_states(states_ego=opp_state, states_opp=ego_state)

        self.ego_planner.set_cost_angles(self.env_options.initial_ego_cost_angle_s,
                                         self.env_options.initial_ego_cost_angle_n)

        self.opp_planner.set_cost_angles(self.env_options.opp_cost_angle_s, 0)

        # Define spaces and stuff related to environment
        self.action_space = spaces.Box(low=np.array([-self.env_options.max_angle,
                                                     -self.env_options.max_angle]),
                                       high=np.array([self.env_options.max_angle,
                                                      self.env_options.max_angle]))

        # Defining the observation space:
        # Followings states are observed:
        # Ego vehicle lateral position n
        # Opposing vehicle lateral position n
        # Distance to opposing vehicle delta_s
        # Velocity difference to opposing vehicle
        # Lateral velocity difference to opposingn vehicle
        maximum_lateral_position = self.ego_planner_options.road_width / 2 - \
                                   self.ego_planner_options.vehicle_radius
        maximum_considered_distance = self.ego_planner_options.maximum_velocity_ego * 10
        self.observation_space = spaces.Box(low=np.array([-maximum_lateral_position,
                                                          -maximum_lateral_position,
                                                          -maximum_considered_distance,
                                                          -self.env_options.maximum_considered_vdiff,
                                                          -self.env_options.maximum_considered_vdiff]),
                                            high=np.array([maximum_lateral_position,
                                                           maximum_lateral_position,
                                                           maximum_considered_distance,
                                                           self.env_options.maximum_considered_vdiff,
                                                           self.env_options.maximum_considered_vdiff]))

        self.reward_range = (-2, 2)
        self.counter = 0
        self.max_iterations = self.env_options.max_iterations

    def step(self, action):
        if action[0] > self.env_options.max_angle:
            action[0] = self.env_options.max_angle
            warnings.warn("Cost angle for s out of range. Clipping.")
        if action[0] < -self.env_options.max_angle:
            action[0] = -self.env_options.max_angle
            warnings.warn("Cost angle for s out of range. Clipping.")
        if action[1] > self.env_options.max_angle:
            action[1] = self.env_options.max_angle
            warnings.warn("Cost angle for n out of range. Clipping.")
        if action[1] < -self.env_options.max_angle:
            action[1] = -self.env_options.max_angle
            warnings.warn("Cost angle for n out of range. Clipping.")
        self.ego_planner.set_cost_angles(action[0], action[1])

        self.ego_planner.solve()
        if self.full_ego_trajectory.size == 0:
            self.full_ego_trajectory = self.ego_planner.get_states_until_time(
                (self.env_options.index_of_simulated_steps - 1) * self.nlp_options.time_disc)
            self.full_ego_trajectory = np.expand_dims(self.full_ego_trajectory, axis=2)
        else:
            return_traj = self.ego_planner.get_states_until_time(
                (self.env_options.index_of_simulated_steps - 1) * self.nlp_options.time_disc)
            full_traj = np.expand_dims(return_traj, axis=2)
            self.full_ego_trajectory = np.append(self.full_ego_trajectory, full_traj, axis=2)
        ego_state = self.ego_planner.get_state_at_time(
            (self.env_options.index_of_simulated_steps) * self.nlp_options.time_disc)

        self.opp_planner.solve()
        if self.full_opp_trajectory.size == 0:
            self.full_opp_trajectory = self.opp_planner.get_states_until_time(
                (self.env_options.index_of_simulated_steps - 1) * self.nlp_options.time_disc)
            self.full_opp_trajectory = np.expand_dims(self.full_opp_trajectory, axis=2)
        else:
            return_traj = self.opp_planner.get_states_until_time(
                (self.env_options.index_of_simulated_steps - 1) * self.nlp_options.time_disc)
            full_traj = np.expand_dims(return_traj, axis=2)
            self.full_opp_trajectory = np.append(self.full_opp_trajectory, full_traj, axis=2)
        opp_state = self.opp_planner.get_state_at_time(
            (self.env_options.index_of_simulated_steps) * self.nlp_options.time_disc)
        self.ego_planner.set_states(ego_state, opp_state)
        self.opp_planner.set_states(opp_state, ego_state)

        # Followings states are observed:
        # Ego vehicle lateral position n
        # Opposing vehicle lateral position n
        # Distance to opposing vehicle delta_s
        # Velocity difference to opposing vehicle
        ego_pos_lat = ego_state[2]
        opp_pos_lat = opp_state[2]
        dist_vehicles = ego_state[0] - opp_state[0]
        diff_long_vel = np.clip(ego_state[1] - opp_state[1], -self.env_options.maximum_considered_vdiff,
                                self.env_options.maximum_considered_vdiff)
        diff_lat_vel = np.clip(ego_state[3] - opp_state[3], -self.env_options.maximum_considered_vdiff,
                               self.env_options.maximum_considered_vdiff)

        observation = np.array([ego_pos_lat, opp_pos_lat, dist_vehicles, diff_long_vel, diff_lat_vel])

        self.counter += 1
        done = False
        if self.counter >= self.max_iterations:
            done = True

        # reward function
        if ego_state[0] > opp_state[0]:
            reward = 1
        else:
            reward = 0
        # reward += ego_state[0]/30
        reward += -np.abs(opp_state[1] + opp_state[3]) / 3
        # add reward for actions differing from driving straight
        reward += (np.sign(action[0]) + action[1] / (np.pi / 4)) * 1e-4

        info = "Void"
        return observation, reward, done, info

    # Execute one time step within the environment
    def reset(self):
        self.counter = 0
        self.ego_planner.set_cost_angles(self.env_options.initial_ego_cost_angle_s,
                                         self.env_options.initial_ego_cost_angle_n)

        self.opp_planner.set_cost_angles(self.env_options.opp_cost_angle_s, 0)

        ego_state = np.array([self.env_options.initial_pos_lon_ego,
                              self.env_options.initial_velocity_ego,
                              self.env_options.initial_pos_lat_ego,
                              0])
        opp_state = np.array([self.env_options.initial_pos_lon_ego + self.env_options.initial_distance_long,
                              self.env_options.initial_velocity_opposing,
                              self.env_options.initial_pos_lat_ego + self.env_options.initial_distance_lat,
                              0])
        self.ego_planner.set_states(ego_state, opp_state)
        self.opp_planner.set_states(opp_state, ego_state)
        self.full_ego_trajectory = np.array([])  # size states/indexes/iterates
        self.full_opp_trajectory = np.array([])

        # Get initial state for return
        ego_pos_lat = ego_state[2]
        opp_pos_lat = opp_state[2]
        dist_vehicles = ego_state[0] - opp_state[0]
        diff_long_vel = np.clip(ego_state[1] - opp_state[1], -self.env_options.maximum_considered_vdiff,
                                self.env_options.maximum_considered_vdiff)
        diff_lat_vel = np.clip(ego_state[3] - opp_state[3], -self.env_options.maximum_considered_vdiff,
                               self.env_options.maximum_considered_vdiff)

        observation = np.array([ego_pos_lat, opp_pos_lat, dist_vehicles, diff_long_vel, diff_lat_vel])

        return observation

    # Reset the state of the environment to an initial state
    def render(self, mode='human', close=False):
        (_, _, overall_iterations) = self.full_ego_trajectory.shape
        fig, ax = plt.subplots()
        max_x_value = np.max([np.max(self.full_ego_trajectory[0, :, :]), np.max(self.full_opp_trajectory[0, :, :])])
        min_x_value = np.min([np.min(self.full_ego_trajectory[0, :, :]), np.min(self.full_opp_trajectory[0, :, :])])
        ax.set_xlim((min_x_value, max_x_value))
        ax.set_ylim((-self.ego_planner_options.road_width, self.ego_planner_options.road_width))

        cmap1 = plt.get_cmap('winter')
        cmap2 = plt.get_cmap('autumn')
        colors1 = [cmap1(i) for i in np.linspace(0, 1, overall_iterations)]
        colors2 = [cmap2(i) for i in np.linspace(0, 1, overall_iterations)]

        for current_run in range(overall_iterations):
            x_ego = self.full_ego_trajectory[:, :, current_run]
            x_oppo = self.full_opp_trajectory[:, :, current_run]
            for i in range(len(x_ego[0, :])):
                circle1 = plt.Circle((x_ego[0, i], x_ego[2, i]),
                                     self.ego_planner_options.vehicle_radius,
                                     fill=False,
                                     color=colors1[current_run])
                ax.add_patch(circle1)
            for i in range(len(x_ego[0, :])):
                circle1 = plt.Circle((x_oppo[0, i],
                                      x_oppo[2, i]),
                                     self.ego_planner_options.vehicle_radius,
                                     color=colors1[current_run],
                                     fill=False)
                ax.add_patch(circle1)
        plt.axhline(y=self.ego_planner_options.road_width / 2, color='r', linestyle='-')
        plt.axhline(y=-self.ego_planner_options.road_width / 2, color='r', linestyle='-')
        plt.show()

    def render_animated(self, mode='human', close=False):
        (n_state, n_traj_states, overall_iterations) = self.full_ego_trajectory.shape
        fig, ax = plt.subplots()
        # fig.tight_layout()

        funargs = (ax, self.full_ego_trajectory, self.full_opp_trajectory, self.ego_planner_options.vehicle_radius,
                   self.ego_planner_options.road_width)
        ani = FuncAnimation(fig, animate, fargs=funargs,
                            interval=self.nlp_options.time_disc * self.env_options.sim_speed,
                            frames=n_traj_states * overall_iterations,
                            repeat=True,
                            repeat_delay=1000)

        plt.show()


if __name__ == "__main__":
    print("Test race cars")
    race_car_env = RaceCarEnv()
    done = False
    episodes = 1
    for cnt_epi in range(episodes):
        race_car_env.reset()
        while done != True:
            angle_s = np.random.normal(np.pi / 4, np.pi / 32, 1)
            angle_n = np.random.normal(0, np.pi / 16, 1)
            action = [angle_s, angle_n]
            observation, reward, done, info = race_car_env.step(action)
            print("Reward: {}".format(reward))
        done = False

    # race_car_env.render()
    race_car_env.render_animated()

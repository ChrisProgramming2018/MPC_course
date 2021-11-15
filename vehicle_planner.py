import casadi as cs
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import Any, List, Dict
from scipy import interpolate
import math


@dataclass
class VehiclePlannerOptions:
    road_width: float = 10.  # Time steps for optimization horizon
    maximum_velocity_ego: float = 5  # Maximum velocity of the ego vehicle
    maximum_velocity_opposing: float = 20  # Maximum velocity of the opposing vehicle
    maximum_lateral_acc: float = 1  # Maximum lateral acceleration for both cars
    maximum_acc: float = 2  # Maximum acceleration for both cars
    air_drag_force: float = 1  # Constant negative air drag decelerating force
    vehicle_mass: float = 1
    vehicle_radius: float = 1.9  # circular vehicle shapes
    weight_inputs: np.ndarray = np.array([1e-1, 1e-1])
    maximum_dec: float = 3.  # Maximum deceleration of ego vehicle. Use positive values.


@dataclass
class NlpOptions:
    n_nodes: int = 10  # Time steps for optimization horizon
    time_disc: float = 0.2
    cost_angle_s: float = np.pi / 4
    cost_angle_n: float = 0
    cost_center_n: float = 1e-1
    print_level_casadi: int = 0
    print_time_ipopt: int = 0
    max_iter: int = 5000
    cost_state_s: float = 2e5
    cost_state_n: float = 1e0
    slack_weight_max_lat_acc: float = 1e5
    slack_weight_obstacle: float = 1e8
    slack_weight_x0: float = 1e8
    slack_weight_max_acc: float = 1e5
    slack_weight_max_dec: float = 1e5


class VehiclePlanner:
    def __init__(self, options: VehiclePlannerOptions, nlp_options: NlpOptions):
        self.road_width = options.road_width  # Road width in meters
        self.maximum_velocity_ego = options.maximum_velocity_ego  # Maximum velocity of the ego vehicle
        self.maximum_velocity_opposing = options.maximum_velocity_opposing  # Maximum velocity of the opposing vehicle
        self.maximum_lateral_acc = options.maximum_lateral_acc  # Maximum lateral acceleration for both cars
        self.maximum_acc = options.maximum_acc  # Maximum acceleration for both cars
        self.air_drag_force = options.air_drag_force  # Constant negative air drag decelerating force
        self.vehicle_mass = options.vehicle_mass
        self.vehicle_radius = options.vehicle_radius  # circular vehicle shapes
        self.weight_inputs = options.weight_inputs
        self.maximum_dec = options.maximum_dec
        self.cost_state_s = nlp_options.cost_state_s
        self.cost_state_n = nlp_options.cost_state_n

        self.n_nodes = nlp_options.n_nodes
        self.time_disc = nlp_options.time_disc
        self.time_pred = (self.n_nodes - 1) * self.time_disc

        self.cost_angle_s = nlp_options.cost_angle_s
        self.cost_angle_n = nlp_options.cost_angle_n
        self.sol = None
        # define vehicle model and solver in casadi
        x_states = cs.MX.sym('x_states', 4)
        u_controls = cs.MX.sym('u_controls', 2)
        x_states_opposing = cs.MX.sym('x_states_opp_veh', 4)

        s = x_states[0]
        vs = x_states[1]
        n = x_states[2]
        vn = x_states[3]

        s_opp = x_states_opposing[0]
        vs_opp = x_states_opposing[1]
        n_opp = x_states_opposing[2]
        vn_opp = x_states_opposing[3]

        u_as = u_controls[0]
        u_an = u_controls[1]

        # dx/dt = f(x,u)
        rhs = cs.vertcat(
            vs,
            u_as - self.air_drag_force,  # - 2/np.pi*cs.atan(vs)*self.air_drag_force/self.vehicle_mass,
            vn,
            u_an)
        f_dyn = cs.Function('f', [x_states, u_controls], [rhs])

        dist_vehicles_sqr = (s - s_opp) ** 2 + (n - n_opp) ** 2
        f_dist_sqr = cs.Function("f_dist_sqr", [x_states, x_states_opposing], [dist_vehicles_sqr])
        # Test function

        f_integrator = cs.Function('F_euler', [x_states, u_controls],
                                   [x_states + self.time_disc * f_dyn(x_states, u_controls)])

        rhs_opp = cs.vertcat(
            vs_opp,
            0,  # - 2/np.pi*cs.atan(vs)*self.air_drag_force/self.vehicle_mass,
            vn_opp,
            0)
        f_dyn_opp = cs.Function('f', [x_states_opposing], [rhs_opp])
        f_integrator_opp = cs.Function('F_euler', [x_states_opposing],
                                       [x_states_opposing + self.time_disc * f_dyn_opp(x_states_opposing)])

        # intg_options = dict()
        # intg_options["tf"] =  self.time_disc
        # intg = cs.integrator('intg', 'rk', {'x': x_states, 'p': u_controls, 'ode': f(x_states, u_controls)},
        #                  intg_options)
        # res = intg(x0=x_states, p=u_controls)
        # xf = res["xf"]
        # f_integrator = Function('F', [x_states, u_controls], [xf])

        # optimization problem
        self.opti = cs.Opti()

        options = {'print_time': nlp_options.print_time_ipopt,
                   'ipopt': {'max_iter': nlp_options.max_iter,
                             'print_level': nlp_options.print_level_casadi,
                             'acceptable_tol': 1e-5,
                             'acceptable_obj_change_tol': 1e-5}}
        self.opti.solver("ipopt", options)

        # Formulate the DAE
        self.X = self.opti.variable(4, self.n_nodes)
        self.U = self.opti.variable(2, self.n_nodes - 1)
        self.cost_dx = self.opti.parameter(1)
        self.cost_dy = self.opti.parameter(1)
        self.X0 = self.opti.parameter(4, 1)
        self.X0_veh_opp = self.opti.parameter(4, 1)
        self.deactivate_vel_limit = self.opti.parameter(1)
        self.slacks = self.opti.variable(5)
        self.activate_collision = self.opti.parameter(1)

        S_vec = self.X[0, :]
        VS_vec = self.X[1, :]
        N_vec = self.X[2, :]
        VN_vec = self.X[3, :]

        dVS_vec = self.U[0, :]
        dVN_vec = self.U[1, :]

        self.U_modif = cs.vertcat(dVS_vec * (1 - self.deactivate_vel_limit), dVN_vec)

        R = np.diag(self.weight_inputs)

        self.opti.subject_to((self.X[:, 0] - self.X0) <= self.slacks[3])  # initial condition constraints
        self.opti.subject_to((self.X[:, 0] - self.X0) >= -self.slacks[3])
        self.obj = 0
        # Build up a graph of integrator calls by means of a nonlinear program
        for k in range(self.n_nodes - 1):
            x_next = f_integrator(self.X[:, k], self.U_modif[:, k])
            self.opti.subject_to(self.X[:, k + 1] == x_next)  # close the gaps

            state_s0 = self.X[0, 0]
            state_n0 = self.X[2, 0]
            # Set to k+1, because first state is fixed anyways and range only goes until (N-1)
            state_s = self.X[0, k + 1]
            state_n = self.X[2, k + 1]
            con = self.U[:, k]

            state_cost = self.cost_state_s * (state_s - state_s0) * self.cost_dx + \
                         (state_n - state_n0) * self.cost_dy * self.cost_state_n

            control_cost = cs.mtimes(cs.transpose(con), cs.mtimes(R, con))

            self.obj = self.obj + state_cost + control_cost + state_n ** 2 * nlp_options.cost_center_n

        S = np.diag([nlp_options.slack_weight_obstacle,
                     nlp_options.slack_weight_max_dec,
                     nlp_options.slack_weight_max_acc,
                     nlp_options.slack_weight_x0,
                     nlp_options.slack_weight_max_lat_acc])
        slack_cost = cs.mtimes(cs.transpose(self.slacks), cs.mtimes(S, self.slacks))
        self.obj += slack_cost
        self.opti.subject_to(-(self.road_width / 2 - self.vehicle_radius) <=
                             (N_vec <= self.road_width / 2 - self.vehicle_radius))
        self.opti.subject_to(0 <= (VS_vec <= (self.maximum_velocity_ego
                                              + (self.deactivate_vel_limit * 1e6))))
        self.opti.subject_to(
            -self.maximum_lateral_acc - self.slacks[4] <= (dVN_vec <= self.maximum_lateral_acc + self.slacks[4]))
        self.opti.subject_to(-self.maximum_dec - self.slacks[1] <= (dVS_vec <= (self.maximum_acc + self.slacks[2])))

        x_opposing = self.X0_veh_opp
        self.x_opposing_container = [x_opposing]
        for k in range(self.n_nodes):
            self.opti.subject_to(
                (self.vehicle_radius * 2) ** 2 - self.slacks[0] - 1e3 * (1 - self.activate_collision) <= f_dist_sqr(
                    self.X[:, k], x_opposing))
            x_opposing = f_integrator_opp(x_opposing)
            self.x_opposing_container.append(x_opposing)
        self.opti.minimize(self.obj)

    def solve(self):
        self.sol = self.opti.solve()
        self.x_full = self.sol.value(self.X)
        self.u_full = self.sol.value(self.U)

    def get_trajectory(self):
        return self.x_full

    def get_state_at_time(self, relative_time: float) -> np.ndarray:
        assert (relative_time <= self.time_pred)
        assert (relative_time >= 0)
        div_rest = math.fmod(relative_time, self.time_disc)
        if abs(div_rest) < 1e-6 or abs(div_rest - self.time_disc) < 1e-6:
            index = int(relative_time / self.time_disc)
            ego_state = self.x_full[:, index]
        else:
            t = np.arange(0, self.time_pred + 1e-6, self.time_disc)
            fx = interpolate.interp1d(t, self.x_full)
            ego_state = fx(relative_time)
        return ego_state

    def get_states_until_time(self, end_time: float, sample_time: float = -0.1):
        if sample_time < 0:
            sample_time = self.time_disc  # Hackmack
        assert (end_time <= self.time_pred)
        assert (end_time >= 0)
        div_rest = math.fmod(end_time, sample_time)
        assert (abs(div_rest) < 1e-6 or abs(div_rest - sample_time) < 1e-6)

        if abs(sample_time - self.time_disc) < 1e-6:
            index = int(end_time / self.time_disc) + 1
            ego_states = self.x_full[:, 0:index]
        else:
            t = np.arange(0, self.time_pred, self.time_disc)
            fx = interpolate.interp1d(t, self.x_full)
            t_out = np.arange(0, end_time + sample_time, sample_time)
            ego_states = fx(t_out)
        return ego_states

    def get_opposing_trajectory(self):
        traj = []
        for i in range(self.x_full.shape[1]):
            traj.append(self.sol.value(self.x_opposing_container[i]))
        return np.array(traj).transpose()

    def set_states(self, states_ego: np.ndarray, states_opp: np.ndarray):
        epsilon = 0.5
        if states_ego[1] > self.maximum_velocity_ego + epsilon:
            deactivate_vel_lim = 1
        else:
            deactivate_vel_lim = 0
        if states_ego[0] < states_opp[0]:
            activate_collision = 1
        else:
            activate_collision = 0
        self.opti.set_value(self.activate_collision, activate_collision)
        self.opti.set_value(self.deactivate_vel_limit, deactivate_vel_lim)
        self.opti.set_value(self.X0, states_ego)
        self.opti.set_value(self.X0_veh_opp, states_opp)

    def set_cost_angles(self, cost_s_angle: float, cost_n_angle: float):
        cost_dx = -np.sin(cost_s_angle)
        cost_dy = -np.sin(cost_n_angle)
        self.opti.set_value(self.cost_dx, cost_dx)
        self.opti.set_value(self.cost_dy, cost_dy)


if __name__ == "__main__":
    print("Test race cars")
    print("--------------")

    vehicle_planner_options = VehiclePlannerOptions()
    nlp_options = NlpOptions()
    nlp_options.n_nodes = 10
    nlp_options.cost_center_n = 0.01
    vehicle_planner = VehiclePlanner(options=vehicle_planner_options, nlp_options=nlp_options)

    vehicle_planner.set_states(np.array([0, 5, 2, 0]), np.array([0, 0.5, -2, 0]))
    vehicle_planner.set_cost_angles(np.pi / 4, (-np.pi / 8))
    vehicle_planner.solve()
    x_ego = vehicle_planner.get_trajectory()
    x_oppo = vehicle_planner.get_opposing_trajectory()

    fig, ax = plt.subplots()
    ax.set_xlim((0, max(x_ego[0, :])))
    ax.set_ylim((-vehicle_planner_options.road_width, vehicle_planner_options.road_width))
    for i in range(len(x_ego[0, :])):
        circle1 = plt.Circle((x_ego[0, i], x_ego[2, i]), vehicle_planner_options.vehicle_radius, fill=False)
        ax.add_patch(circle1)
    for i in range(len(x_ego[0, :])):
        circle1 = plt.Circle((x_oppo[0, i], x_oppo[2, i]), vehicle_planner_options.vehicle_radius, color='r',
                             fill=False)
        ax.add_patch(circle1)
    plt.axhline(y=vehicle_planner_options.road_width / 2, color='r', linestyle='-')
    plt.axhline(y=-vehicle_planner_options.road_width / 2, color='r', linestyle='-')
    print("State at time: ")
    print(vehicle_planner.get_state_at_time(1.1))
    print(vehicle_planner.get_states_until_time(1.2))
    plt.show()

from matplotlib import pyplot as plt
import numpy as np


# Animation function
def animate(i, ax, x_ego_full: np.ndarray, x_oppo_full: np.ndarray, vehicle_radius: float, road_width: float):
    (n_states, n_trajectroy_states, overall_iterations) = x_ego_full.shape
    x_ego_flat = x_ego_full.transpose(0, 2, 1).reshape(n_states, n_trajectroy_states * overall_iterations)
    x_oppo_flat = x_oppo_full.transpose(0, 2, 1).reshape(n_states, n_trajectroy_states * overall_iterations)
    x_ego = x_ego_flat[:, i]
    x_oppo = x_oppo_flat[:, i]
    if i % n_trajectroy_states == 0:
        print("--------------------New MPC Simulation---------------------------")
    print("Velocity Ego: {} m/s. Oppo: {} m/s".format(x_ego[1], x_oppo[1]))

    ax.clear()
    ax.axis('equal')
    ax.set(xlim=(np.minimum(x_ego[0], x_oppo[0]) - 20.,
                 np.maximum(x_ego[0], x_oppo[0]) + 20.),
           ylim=(-road_width,
                 road_width))

    circle1 = plt.Circle((x_ego[0], x_ego[2]),
                         vehicle_radius,
                         fill=True,
                         color="mediumseagreen")
    ax.add_patch(circle1)
    circle1 = plt.Circle((x_oppo[0], x_oppo[2]),
                         vehicle_radius,
                         fill=True,
                         color="steelblue")
    ax.add_patch(circle1)
    ax.text(x_ego[0] - 1, x_ego[2] - 0.5, "MPCRL\nEgo", fontsize=10, color="white")
    ax.text(x_oppo[0] - 1, x_oppo[2] - 0.5, "MPC\nOpp", fontsize=10, color="white")

    ax.axhline(y=road_width / 2, color='grey', linestyle='-', linewidth=2)
    ax.axhline(y=-road_width / 2, color='grey', linestyle='-', linewidth=2)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.set_axisbelow(True)

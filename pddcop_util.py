import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
import gc
import math
import itertools
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d


def lowercase(out_type):
  if out_type == "INFINITE_HORIZON":
    return "infinite"
  elif out_type == "FINITE_HORIZON":
    return "finite"

  return ""


def alg(pddcop_alg, dcop_alg):
  if pddcop_alg == "LS_SDPOP" and dcop_alg == "DPOP":
    return "LS-SDPOP"
  elif pddcop_alg == "C_DCOP" and dcop_alg == "DPOP":
    return "C-DPOP"
  # LS-RAND
  elif pddcop_alg == "LS_RAND":
    return "LS-RAND"
  elif pddcop_alg == "LS_SDPOP":
    return "LS-" + dcop_alg
  elif pddcop_alg == "FORWARD":
    return "F-" + dcop_alg
  elif pddcop_alg == "BACKWARD":
    return "B-" + dcop_alg
  else:
    return "C-" + dcop_alg


def online_alg(pddcop_alg, dcop_alg, topology):
  return (pddcop_alg + "_" + dcop_alg + "_" + topology).lower()


def instance_file_online(dynamic_type, pdcop_algorithm, dcop_algorithm, instance_id, decision, random, dx, dy,
                         switching_cost, horizon, discount_factor, heuristic_weight, online_run, topology):
  output_file = "output_files/"
  output_file += topology + "/"
  output_file += dynamic_type + "/"
  output_file += pdcop_algorithm + "_" + dcop_algorithm + "/"
  output_file += "instanceID=" + str(instance_id)
  output_file += "_x=" + str(decision)
  output_file += "_y=" + str(random)
  output_file += "_dx=" + str(dx)
  output_file += "_dy=" + str(dy)
  output_file += "_sw=" + str(switching_cost)
  output_file += "_h=" + str(horizon)
  output_file += "_discountFactor=" + str(discount_factor)
  output_file += "_heuristicWeight=" + str(round(heuristic_weight, 1))
  output_file += "_" + pdcop_algorithm
  output_file += "_" + dcop_algorithm
  output_file += "_" + dynamic_type
  output_file += ".txt"

  return output_file


def instance_file_iteration(dynamic_type, pdcop_algorithm, dcop_algorithm, instance_id, decision, random, dx, dy, switching_cost,
                  horizon, discount_factor, heuristic_weight):
  output_file = "output_files/"
  output_file += dynamic_type + "/"
  output_file += pdcop_algorithm + "_" + dcop_algorithm + "/"
  output_file += "instanceID=" + str(instance_id)
  output_file += "_x=" + str(decision)
  output_file += "_y=" + str(random)
  output_file += "_dx=" + str(dx)
  output_file += "_dy=" + str(dy)
  output_file += "_sw=" + str(switching_cost)
  output_file += "_h=" + str(horizon)
  output_file += "_discountFactor=" + str(discount_factor)
  output_file += "_heuristicWeight=" + str(round(heuristic_weight, 1))
  output_file += "_" + pdcop_algorithm
  output_file += "_" + dcop_algorithm
  output_file += "_" + dynamic_type
  output_file += ".txt"

  return output_file


def instance_file(dynamic_type, pdcop_algorithm, dcop_algorithm, decision, random, dx, dy, switching_cost,
                  horizon, discount_factor, heuristic_weight):
  output_file = "output_files/"
  output_file += dynamic_type + "/"
  output_file += "x=" + str(decision)
  output_file += "_y=" + str(random)
  output_file += "_dx=" + str(dx)
  output_file += "_dy=" + str(dy)
  output_file += "_sw=" + str(switching_cost)
  output_file += "_h=" + str(horizon)
  output_file += "_discountFactor=" + str(discount_factor)
  output_file += "_heuristicWeight=" + str(round(heuristic_weight, 1))
  output_file += "_" + pdcop_algorithm
  output_file += "_" + dcop_algorithm
  output_file += "_" + dynamic_type
  output_file += ".txt"

  return output_file


def effective_reward(result_instance, pdcop_algorithm, horizon, time_duration, dcop_algorithm, topology):
  # MGM has minus infinity
  if topology == "random":
    if dcop_algorithm == "DPOP":
      first_time_step = 1
    elif dcop_algorithm == "MGM":
      first_time_step = 3
  else:
    # MGM has minus infinity
    if topology == "meeting":
      if dcop_algorithm == "DPOP":
        # React might have infinity rewards by randomize solutions for meeting scheduling problems
        first_time_step = 2
      elif dcop_algorithm == "MGM":
        print(result_instance)
        first_time_step = 3

  qualities = result_instance.iloc[first_time_step - 1:, 1].astype(float)
  costs = result_instance.iloc[first_time_step - 1:, 2].astype(int)
  solve_times = result_instance.iloc[first_time_step - 1:, 3].astype(int)
  reward = 0

  # for time_step in range(1, horizon + 2):
  for time_step in range(first_time_step, horizon + 2):
    if pdcop_algorithm == "REACT":
      reward += solve_times[time_step] * qualities[time_step - 1] + (
        time_duration - solve_times[time_step]) * qualities[time_step] - time_duration * costs[time_step]
    else:
      reward += time_duration * (qualities[time_step] - costs[time_step])

  return reward


def plot_3d(time_durations, switching_costs, pdcop_algorithm, dcop_algorithm, topology):
  print(time_durations, switching_costs, pdcop_algorithm, dcop_algorithm, topology)
  online_average = "online_average_"
  fig_size = (10, 6)
  tick_size = 10
  label_size = 10

  azimuth, elevation = get_3d_par(dcop_algorithm, topology)
  # zticks, z_precision = get_z_par(dcop_algorithm, topology)

  result_df = pd.read_csv(online_average + online_alg(pdcop_algorithm, dcop_algorithm, topology) + ".csv", index_col=0)
  fig = plt.figure(figsize=fig_size)
  ax = fig.add_subplot(111, projection='3d')
  xx, yy = np.meshgrid(time_durations, switching_costs)
  print(sign_log(result_df.to_numpy()))
  # ax.plot_surface(xx, yy, np.log(result_df.to_numpy()), cmap=color_map(pdcop_algorithm), edgecolor='none')
  ax.plot_surface(xx, yy, result_df.to_numpy(), cmap=color_map(pdcop_algorithm), edgecolor='none')
  # ax.plot_surface(xx, yy, sign_log(result_df.to_numpy()), cmap=color_map(pdcop_algorithm), edgecolor='none')
  ax.set_xticks([time_durations[i] for i in range(0, len(time_durations), 4)])
  ax.set_xticklabels([time_durations[i] for i in range(0, len(time_durations), 4)], fontsize=tick_size)
  ax.set_yticks(switching_costs)
  ax.set_yticklabels(switching_costs, fontsize=tick_size)
  # ax.set_zticks(zticks)
  # ax.set_zticklabels([z_precision % x for x in zticks], fontsize=tick_size)
  ax.set_xlabel("Time Duration", fontsize=label_size)
  ax.set_ylabel("Switching Cost", fontsize=label_size)
  ax.set_zlabel("Difference in Effective Utilities", fontsize=label_size)
  ax.view_init(elev=elevation, azim=azimuth)
  plt.savefig(online_average + online_alg(pdcop_algorithm, dcop_algorithm, topology) + ".pdf", bbox_inches='tight')
  # plt.show()
  print("PLOTTING=====" + online_average + online_alg(pdcop_algorithm, dcop_algorithm, topology) + ".pdf")
  plt.cla()
  plt.clf()
  plt.close('all')
  gc.collect()


def color_map(pddcop_algorithm):
  if pddcop_algorithm == "FORWARD":
    return "Blues"
  elif pddcop_algorithm == "HYBRID":
    return "Greens"

# Return azimuth, elevation
def get_3d_par(dcop_algorithm, instance):
  print(dcop_algorithm, instance)
  if (dcop_algorithm, instance) == ("DPOP", "random"):
    return -48, 12

  return -48, 12


# return zticks
def get_z_par(dcop_algorithm, instance):
  if (dcop_algorithm, instance) == ("DPOP", "random"):
    return np.arange(16.8, 17.01, 0.05), "%.2f"
  if (dcop_algorithm, instance) == ("DPOP", "meeting"):
    return np.arange(11, 15.01, 1), "%.0f"

  return np.arange(16.8, 17.01, 0.05)


def sign_log(value_ndarray):
  for i, x in np.ndenumerate(value_ndarray):
    if x > 0:
      value_ndarray[i] = np.log(x)
    else:
      value_ndarray[i] = -np.log(np.abs(x))

  return value_ndarray


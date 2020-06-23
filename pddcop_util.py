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


def instance_file_online(dynamic_type, pdcop_algorithm, dcop_algorithm, instance_id, decision, random, dx, dy,
                         switching_cost, horizon, discount_factor, heuristic_weight, online_run):
  output_file = "output_files/"
  output_file += dynamic_type + "/"
  output_file += pdcop_algorithm + "_" + dcop_algorithm + "/"
  output_file += "OnlineRun=" + str(online_run)
  output_file += "_instanceID=" + str(instance_id)
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

def effective_reward(result_instance, pdcop_algorithm, horizon, time_duration):
  qualities = result_instance.iloc[:, 1].astype(float)
  costs = result_instance.iloc[:, 2].astype(int)
  solve_times = result_instance.iloc[:, 3].astype(int)
  reward = 0
  for time_step in range(1, horizon + 2):
    if pdcop_algorithm == "REACT":
      reward += solve_times[time_step] * qualities[time_step - 1] + (
        time_duration - solve_times[time_step]) * qualities[time_step] - time_duration * costs[time_step]
    else:
      reward += time_duration * (qualities[time_step] - costs[time_step])

  return reward


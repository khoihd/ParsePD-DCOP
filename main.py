from pddcop_util import *


def generate_table_meeting():
  dx = 5
  dy = 5
  horizon = 4
  switching_cost = 50
  algorithms = [("LS_SDPOP", "DPOP"), ("LS_SDPOP", "MGM"), ("LS_RAND", "DPOP"),
                ("FORWARD", "DPOP"), ("FORWARD", "MGM"),
                ("BACKWARD", "DPOP"), ("BACKWARD", "MGM")
                ]
  # algorithms = [("LS_RAND", "DPOP")]
  dynamic_types = ["FINITE_HORIZON", "INFINITE_HORIZON"]
  # dynamic_types = ["INFINITE_HORIZON"]
  switching_costs = 50
  discount_factor = 0.9
  instances = range(30)
  result = {}
  for dynamic_type in dynamic_types:
    df = pd.DataFrame(columns=algorithms)
    for decision in range(4, 11, 2):
      random = int(decision / 2)
      for (pdcop_algorithm, dcop_algorithm) in algorithms:
        if (pdcop_algorithm, dcop_algorithm) == ("LS_SDPOP", "DPOP"):
          heuristic_weight = 0.6
        else:
          heuristic_weight = 0.0

        output_file = instance_file(dynamic_type, pdcop_algorithm, dcop_algorithm, decision, random, dx, dy,
                                      switching_cost, horizon, discount_factor, heuristic_weight)
        result_instance = pd.read_csv(output_file, delimiter="\t")
        satisfied_instance = 0
        runtimes = []
        for instance_id in range(len(result_instance["Utility"])):
          quality = result_instance["Utility"][instance_id]
          runtime = result_instance["Time (ms)"][instance_id]
          if quality == "-∞":
            print("Negative Infinity")
            None
          elif float(quality) < 0:
            print("Negative")
            None
          elif math.isnan(float(quality)):
            print("Not a Number")
            None
          else:
            satisfied_instance = satisfied_instance + 1
            runtimes.append(runtime)
        if satisfied_instance > 0:
          df.loc[decision, (pdcop_algorithm, dcop_algorithm)] = (
            int(satisfied_instance / len(range(30)) * 100), int(np.average(runtimes)))
        else:
          df.loc[decision, (pdcop_algorithm, dcop_algorithm)] = (0, 0)
    result[dynamic_type] = df
    print(dynamic_type)
    print(df)
    df.to_csv("meeting_" + lowercase(dynamic_type) + ".csv")


def plot_online_3d():
  decision = 10
  random = 10
  dx = 5
  dy = 5
  discount_factor = 0.9
  horizon = 10
  heuristic_weight = 0.0
  dynamic_type = "ONLINE"

  pddcop_algorithms = ["FORWARD", "HYBRID"]
  # dcop_algorithms = ["DPOP"]
  dcop_algorithms = ["MGM"]

  # topologies = ["random", "meeting"]
  topologies = ["random"]
  switching_costs = range(0, 11, 2)
  time_durations = range(2000, 4001, 100)
  instances = range(30)

  online_average = "online_average_"
  online_max = "online_max_"
  for pdcop_algorithm, dcop_algorithm, topology in itertools.product(pddcop_algorithms, dcop_algorithms, topologies):
    # break

    average_diff = pd.DataFrame()
    max_diff = pd.DataFrame()
    for switching_cost in switching_costs:
      for time_duration in time_durations:
        print(switching_cost, time_duration, dcop_algorithm, pdcop_algorithm, topology)
        diff_eff = []
        for instance_id in instances:
          proactive_avg = []
          reactive_avg = []
          for online_run in range(1):
            # Get file name
            proactive_file = instance_file_online(dynamic_type, pdcop_algorithm, dcop_algorithm, instance_id, decision,
                                                  random, dx, dy, switching_cost, horizon, discount_factor,
                                                  heuristic_weight, online_run, topology)
            reactive_file = instance_file_online(dynamic_type, "REACT", dcop_algorithm, instance_id, decision, random,
                                                 dx, dy, switching_cost, horizon, discount_factor, heuristic_weight,
                                                 online_run, topology)
            # Read CSV instance
            proactive_instance = pd.read_csv(proactive_file, delimiter="\t")
            reactive_instance = pd.read_csv(reactive_file, delimiter="\t")
            # Compute Effective Reward
            proactive_eff_reward = effective_reward(proactive_instance, pdcop_algorithm, horizon, time_duration, dcop_algorithm, topology)
            reactive_eff_reward = effective_reward(reactive_instance, "REACT", horizon, time_duration, dcop_algorithm, topology)
            # Append to the list to compute average later
            proactive_avg.append(proactive_eff_reward)
            reactive_avg.append(reactive_eff_reward)
          diff_eff.append(np.average(proactive_eff_reward) - np.average(reactive_eff_reward))
        average_diff.loc[switching_cost, time_duration] = np.average(diff_eff)
        max_diff.loc[switching_cost, time_duration] = np.max(diff_eff)
    # print(online_average + online_alg(pdcop_algorithm, dcop_algorithm) + ".csv")
    average_diff.to_csv(online_average + online_alg(pdcop_algorithm, dcop_algorithm, topology) + ".csv")
    max_diff.to_csv(online_max + online_alg(pdcop_algorithm, dcop_algorithm, topology) + ".csv")

  for pdcop_algorithm, dcop_algorithm, topology in itertools.product(pddcop_algorithms, dcop_algorithms, topologies):
    plot_3d(time_durations, switching_costs, pdcop_algorithm, dcop_algorithm, topology)



def generate_table_agents():
  discount_factor = 0.9
  horizon = 4
  dynamic_types = ["INFINITE_HORIZON", "FINITE_HORIZON"]
  switching_cost = 50
  algorithms = [("C_DCOP", "DPOP"),
                ("LS_SDPOP", "DPOP"), ("LS_SDPOP", "MGM"), ("LS_RAND", "DPOP"),
                ("FORWARD", "DPOP"), ("FORWARD", "MGM"),
                ("BACKWARD", "DPOP"), ("BACKWARD", "MGM")
                ]
  # algorithms = [("LS_SDPOP", "MGM"), ("FORWARD", "MGM"), ("BACKWARD", "MGM")]
  tables = {}
  dx = 3
  dy = 3
  for dynamic_type in dynamic_types:
    df = pd.DataFrame(columns=algorithms)
    for (pdcop_algorithm, dcop_algorithm) in algorithms:
      if (pdcop_algorithm, dcop_algorithm) == ("LS_SDPOP", "DPOP"):
        heuristic_weight = 0.6
      elif (pdcop_algorithm, dcop_algorithm) == ("C_DCOP", "DPOP"):
        heuristic_weight = 0.6
      else:
        heuristic_weight = 0.0
      for decision in range(5, 55, 5):
        output_file = "output_files/"
        output_file += dynamic_type + "/"
        output_file += "x=" + str(decision)
        output_file += "_y=" + str(int(decision / 5))
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

        if not os.path.isfile(output_file):
          continue

        result_instance = pd.read_csv(output_file, delimiter="\t")
        print(result_instance.columns)
        print(output_file)
        quality = round(np.mean(result_instance["Utility"]), 2)
        runtime = round(np.mean(result_instance["Time (ms)"]), 2)
        df.loc[decision, (pdcop_algorithm, dcop_algorithm)] = (quality, runtime)
    tables[dynamic_type] = df
    df.to_csv("agent_" + lowercase(dynamic_type) + ".csv")
  print(tables)


def plot_horizon():
  marker = -1
  dx = 3
  dy = 3
  decision = 5
  random = 1
  switching_cost = 50
  discount_factor = 0.9
  for dynamic_type in ["FINITE_HORIZON", "INFINITE_HORIZON"]:
    fig, ax = plt.subplots()
    for pdcop_algorithm in ["C_DCOP", "FORWARD", "BACKWARD", "LS_RAND", "LS_SDPOP"]:
      for dcop_algorithm in ["DPOP", "MGM"]:
        marker = marker + 1
        if pdcop_algorithm == "C_DCOP" and dcop_algorithm == "DPOP":
          horizons = range(2, 4)
        elif pdcop_algorithm == "C_DCOP" and dcop_algorithm == "MGM":
          continue
        elif pdcop_algorithm == "LS_RAND" and dcop_algorithm == "DPOP":
          continue
        else:
          horizons = range(2, 11)

        if pdcop_algorithm == "LS_SDPOP":
          heuristic_weight = 0.6
        else:
          heuristic_weight = 0.0

        quality = []
        runtime = []
        for horizon in horizons:
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
          result = pd.read_csv(output_file, delimiter="\t")
          quality.append(np.mean(result.iloc[:, 1]))
          runtime.append(np.mean(result.iloc[:, 2]))
          print(pdcop_algorithm + "_" + dcop_algorithm + "_" + dynamic_type)
        ax.plot(horizons, np.log(runtime), marker="s", label=alg(pdcop_algorithm, dcop_algorithm))

    # Plot legends
    fig_legend = plt.figure(figsize=(1.5, 1.3))
    plt.figlegend(*ax.get_legend_handles_labels(), loc='center', ncol=4)
    fig_legend.savefig("horizon_legend.pdf", bbox_inches='tight')
    # End plotting legends

    ax.set_xlabel("Horizon")
    ax.set_ylabel("Runtime (ms) in log scale")
    # plt.legend(loc='best')
    ax.set_yticks(range(2, 16, 2))
    # plt.title(dynamic_type)
    # plt.show()
    fig.savefig("horizon_" + lowercase(dynamic_type) + ".pdf", bbox_inches='tight')
  plt.cla()
  # Clear the current figure.
  plt.clf()
  # Closes all the figure windows.
  plt.close('all')
  gc.collect()


def plot_switching_cost():
  decision = 10
  random = 2
  dx = 3
  dy = 3
  discount_factor = 0.9
  horizon = 4
  dynamic_types = ["FINITE_HORIZON", "INFINITE_HORIZON"]
  algorithms = [("LS_SDPOP", "DPOP"), ("LS_SDPOP", "MGM"), ("LS_RAND", "DPOP")]
  # fig_size = (4, 3)
  for dynamic_type in dynamic_types:
    switching_costs = range(0, 110, 10)
    fig_iteration, ax_iteration = plt.subplots()
    fig_runtime, ax_runtime = plt.subplots()
    fig_quality, ax_quality = plt.subplots()

    for (pdcop_algorithm, dcop_algorithm) in algorithms:
      if (pdcop_algorithm, dcop_algorithm) == ("LS_SDPOP", "DPOP"):
        heuristic_weight = 0.6
      else:
        heuristic_weight = 0.0

      print(dynamic_type, pdcop_algorithm, dcop_algorithm)
      plot_iteration = []
      plot_runtimes = []
      plot_qualities = []
      for switching_cost in switching_costs:
        converged_iterations = []
        converged_runtimes = []
        converged_qualities = []
        for instance_id in range(30):
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
          result_instance = pd.read_csv(output_file, delimiter="\t")
          for row in range(0, result_instance.shape[0] - 1):
            if result_instance["Utility"][row] == result_instance["Utility"][row + 1]:
              converged_iterations.append(result_instance["Iteration"][row] + 1)
              converged_runtimes.append(result_instance["Time (ms)"][row])
              converged_qualities.append(result_instance["Utility"][row])
              break
        plot_iteration.append(np.mean(converged_iterations))
        plot_runtimes.append(np.mean(converged_runtimes))
        plot_qualities.append(np.mean(converged_qualities))
      ax_iteration.plot(switching_costs, plot_iteration, marker="s", label=alg(pdcop_algorithm, dcop_algorithm))
      ax_runtime.plot(switching_costs, plot_runtimes, marker="o", label=alg(pdcop_algorithm, dcop_algorithm))
      ax_quality.plot(switching_costs, plot_qualities, marker="x", label=alg(pdcop_algorithm, dcop_algorithm))
    # Plot switching costs
    ax_iteration.set_xlabel("Switching Cost")
    ax_iteration.set_ylabel("Number of Iterations")
    ax_iteration.set_xticks(switching_costs)
    ax_iteration.set_xticklabels(switching_costs)
    ax_iteration.set_yticks(range(9))
    ax_iteration.set_yticklabels(range(9))
    fig_iteration.savefig("switching_cost_iteration_" + lowercase(dynamic_type) + ".pdf", bbox_inches='tight')

    # Plot runtimes
    ax_runtime.set_xlabel("Switching Cost")
    ax_runtime.set_ylabel("Runtime (ms)")
    ax_runtime.set_xticks(switching_costs)
    ax_runtime.set_xticklabels(switching_costs)
    ax_runtime.set_yticks(range(200, 1400, 200))
    ax_runtime.set_yticklabels(range(200, 1400, 200))
    fig_runtime.savefig("switching_cost_runtime_" + lowercase(dynamic_type) + ".pdf", bbox_inches='tight')

    # Plot runtimes
    ax_quality.set_xlabel("Switching Cost")
    ax_quality.set_ylabel("Solution Quality")
    ax_quality.set_xticks(switching_costs)
    ax_quality.set_xticklabels(switching_costs)
    # ax_runtime.set_yticks(range(200, 1400, 200))
    # ax_runtime.set_yticklabels(range(200, 1400, 200))
    fig_quality.savefig("switching_cost_quality_" + lowercase(dynamic_type) + ".pdf", bbox_inches='tight')

    # Plot legends
    fig_legend = plt.figure(figsize=(1.5, 1.3))
    # Erase maker in handler
    for handler in ax_iteration.get_legend_handles_labels()[0]:
      handler.set_marker(None)
    plt.figlegend(*ax_iteration.get_legend_handles_labels(), loc='center', ncol=3)
    fig_legend.savefig("switching_cost_legend_no_marker.pdf", bbox_inches='tight')
    # End plotting legends
  plt.cla()
  plt.clf()
  plt.close('all')
  gc.collect()


def plot_switching_cost_subplot():
  decision = 10
  random = 2
  dx = 3
  dy = 3
  discount_factor = 0.9
  horizon = 4
  dynamic_types = ["FINITE_HORIZON", "INFINITE_HORIZON"]
  algorithms = [("LS_SDPOP", "DPOP"), ("LS_SDPOP", "MGM"), ("LS_RAND", "DPOP")]
  switching_costs = range(0, 110, 10)

  fig, axs = plt.subplots(2, 2)
  fig.set_figheight(3)
  fig.set_figwidth(10)
  axe_index = 0
  for dynamic_type in dynamic_types:
    for (pdcop_algorithm, dcop_algorithm) in algorithms:
      if (pdcop_algorithm, dcop_algorithm) == ("LS_SDPOP", "DPOP"):
        heuristic_weight = 0.6
      else:
        heuristic_weight = 0.0

      plot_iteration = []
      plot_runtimes = []
      plot_qualities = []
      for switching_cost in switching_costs:
        converged_iterations = []
        converged_runtimes = []
        converged_qualities = []
        for instance_id in range(30):
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
          result_instance = pd.read_csv(output_file, delimiter="\t")
          for row in range(0, result_instance.shape[0] - 1):
            if result_instance["Utility"][row] == result_instance["Utility"][row + 1]:
              converged_iterations.append(result_instance["Iteration"][row] + 1)
              converged_runtimes.append(result_instance["Time (ms)"][row])
              converged_qualities.append(result_instance["Utility"][row])
              break
        plot_iteration.append(np.mean(converged_iterations))
        plot_runtimes.append(np.mean(converged_runtimes))
        plot_qualities.append(np.mean(converged_qualities))

      print(len(plot_iteration), plot_iteration)
      print(len(switching_costs), switching_costs)
      ax_iteration = axs[0 ,axe_index]
      ax_runtime = axs[1, axe_index]
      ax_iteration.plot(switching_costs, plot_iteration, marker="s", label=alg(pdcop_algorithm, dcop_algorithm))
      ax_runtime.plot(switching_costs, plot_runtimes, marker="o", label=alg(pdcop_algorithm, dcop_algorithm))

    # Switching costs
    ax_iteration.set_xlabel("Switching Cost")
    ax_iteration.set_ylabel("Number of Iterations")
    ax_iteration.set_xticks(switching_costs)
    ax_iteration.set_xticklabels(switching_costs)
    ax_iteration.set_yticks(range(9))
    ax_iteration.set_yticklabels(range(9))
    # Runtime
    ax_runtime.set_xlabel("Switching Cost")
    ax_runtime.set_ylabel("Runtime (ms)")
    ax_runtime.set_xticks(switching_costs)
    ax_runtime.set_xticklabels(switching_costs)
    ax_runtime.set_yticks(range(200, 1400, 200))
    ax_runtime.set_yticklabels(range(200, 1400, 200))
    axe_index = axe_index + 1;
  fig.tight_layout()
  fig.savefig("switching_cost.pdf", bbox_inches='tight')

  # Plot legends
  fig_legend = plt.figure(figsize=(1.5, 1.3))
  # Erase maker in handler
  for handler in ax_iteration.get_legend_handles_labels()[0]:
    handler.set_marker(None)
  plt.figlegend(*ax_iteration.get_legend_handles_labels(), loc='center', ncol=3)
  fig_legend.savefig("switching_cost_legend_no_marker.pdf", bbox_inches='tight')
  # End plotting legends
  plt.cla()
  plt.clf()
  plt.close('all')
  gc.collect()


def plot_heuristic():
  dx = 3
  dy = 3
  decision = 10
  random = 2
  horizon = 4
  switching_cost = 50
  discount_factor = 0.9
  for dynamic_type in ["INFINITE_HORIZON", "FINITE_HORIZON"]:
    quality = []
    runtime = []
    for heuristic_weight in np.linspace(0, 1, 11):
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
      output_file += "_" + "LS_SDPOP"
      output_file += "_" + "DPOP"
      output_file += "_" + dynamic_type
      output_file += ".txt"
      result = pd.read_csv(output_file, delimiter="\t")
      quality.append(np.mean(result.iloc[0:29, 1]))
      runtime.append(np.mean(result.iloc[0:29, 2]))
      print(quality)
      print(runtime)
    fig, ax = plt.figure()
    ax.plot(runtime, marker="s")
    ax.set_xlabel("Heuristic Weights")
    ax.set_ylabel("Runtime (ms)")
    # plt.title(dynamic_type)
    ax.set_yticks(range(250, 501, 50))
    # plt.show()
    fig.savefig("heuristic_" + lowercase(dynamic_type) + ".pdf", bbox_inches='tight')
  plt.cla()
  # Clear the current figure.
  plt.clf()
  # Closes all the figure windows.
  plt.close('all')
  gc.collect()


if __name__ == "__main__":
  plot_online_3d()


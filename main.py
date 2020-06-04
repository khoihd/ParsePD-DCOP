import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
from mpl_toolkits import mplot3d


output_folder = "output_files"
dx = 3
dy = 3


def plot_online_3d():
  decision = 12
  random = 3
  discount_factor = 0.9
  horizon = 4
  heuristic_weight = 0.0
  dynamic_type = "ONLINE"
  dcop_algorithms = ["DPOP"]
  switching_costs = range(0, 210, 10)
  time_durations = range(0, 100, 5)

  algorithm_result = {}
  for pdcop_algorithm in ["FORWARD", "REACT", "HYBRID"]:
    for dcop_algorithm in dcop_algorithms:
      df = pd.DataFrame()
      for switching_cost in switching_costs:
        for time_duration in time_durations:
          for instance_id in range(30):
            output_file = output_folder + "/"
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

            if instance_id == 0:
              result_instance = pd.read_csv(output_file, delimiter="\t")
            else:
              result_instance = pd.read_csv(output_file, delimiter="\t", header=None)

            qualities = result_instance.iloc[:, 1]
            costs = result_instance.iloc[:, 2]
            times = result_instance.iloc[:, 3]
            eff_quality = 0
            for time_step in range(1, horizon + 2):
              eff_quality = times[time_step] * qualities[time_step - 1] + (time_duration - times[time_step]) * \
                            qualities[time_step] - time_duration * costs[time_step]
            df.loc[switching_cost, time_duration] = eff_quality
    df.to_csv(pdcop_algorithm + ".csv")
    algorithm_result[pdcop_algorithm] = df

  fig = plt.figure()
  ax = plt.axes(projection='3d')
  xx, yy = np.meshgrid(switching_costs, time_durations)
  # ax.plot3D(xx, yy, zline, 'blue')

  for pdcop_algorithm in ["FORWARD", "HYBRID"]:
    print(algorithm_result[pdcop_algorithm])
    result_df = pd.DataFrame()
    for switching_cost in switching_costs:
      for time_duration in time_durations:
        # print()
        result_df.loc[switching_cost, time_duration] = algorithm_result[pdcop_algorithm].loc[switching_cost, time_duration] - algorithm_result["REACT"].loc[switching_cost, time_duration]
    # print(result_df)


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
        output_file = output_folder + "/"
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
    df.to_csv(dynamic_type + ".csv")
  print(tables)


def plot_switching_cost():
  marker = -1
  decision = 10
  random = 2
  discount_factor = 0.9
  horizon = 4
  dynamic_types = ["FINITE_HORIZON", "INFINITE_HORIZON"]
  # dynamic_types = ["INFINITE_HORIZON"]
  algorithms = [("LS_SDPOP", "DPOP"), ("LS_SDPOP", "MGM"), ("LS_RAND", "MGM")]
  for dynamic_type in dynamic_types:
    switching_costs = range(0, 110, 10)
    f = plt.figure()
    plt.xlabel("Switching Cost")
    plt.ylabel("Iteration")
    for (pdcop_algorithm, dcop_algorithm) in algorithms:
      marker = marker + 1
      if (pdcop_algorithm, dcop_algorithm) == ("LS_SDPOP", "DPOP"):
        heuristic_weight = 0.6
      else:
        heuristic_weight = 0.0

      plot_iteration = []
      for switching_cost in switching_costs:
        converged_iterations = []
        for instance_id in range(0, 30, 1):
          output_file = output_folder + "/"
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
              break
        plot_iteration.append(np.mean(converged_iterations))
      print(len(plot_iteration), plot_iteration)
      print(len(switching_costs), switching_costs)
      plt.plot(switching_costs, plot_iteration, marker="s", label=pdcop_algorithm + "_" + dcop_algorithm)
    plt.legend(loc='best')
    plt.title("Varying switching costs " + dynamic_type)
    plt.xticks(switching_costs)
    plt.show()
    plt.savefig("switching_cost" + dynamic_type + ".pdf", bbox_inches='tight')


def plot_horizon():
  marker = -1
  decision = 10
  random = 1
  switching_cost = 50
  discount_factor = 0.9
  for dynamic_type in ["FINITE_HORIZON"]:
    for pdcop_algorithm in ["C_DCOP", "FORWARD", "BACKWARD", "LS_RAND", "LS_SDPOP"]:
      for dcop_algorithm in ["DPOP", "MGM"]:
        marker = marker + 1
        if pdcop_algorithm == "C_DCOP" and dcop_algorithm == "DPOP":
          horizons = range(2, 4)
          f = plt.figure()
          plt.xlabel("Horizon")
          plt.ylabel("Runtime (ms)")
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
          output_file = output_folder + "/"
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
          print(pdcop_algorithm + "_" + dcop_algorithm + "_" + str(runtime))
        plt.plot(horizons, np.log(runtime), marker=marker, label=pdcop_algorithm + "_" + dcop_algorithm)

    plt.legend(loc='best')
    plt.yticks(range(2, 16, 2))
    # plt.yticks(range(0, 1100, 100))
    plt.title("Varying horizon " + dynamic_type)
    plt.show()
    plt.savefig("varying_horizon_" + dynamic_type + ".pdf", bbox_inches='tight')


def plot_heuristic():
  quality = []
  runtime = []
  for pdcop_algorithm in ["LS_SDPOP"]:
    for dcop_algorithm in ["DPOP"]:
      for decision in [10]:
        for random in [2]:
          for horizon in [4]:
            for switching_cost in [50]:
              for discount_factor in [0.9]:
                for dynamic_type in ["INFINITE_HORIZON"]:
                  for heuristic_weight in np.linspace(0, 1, 11):
                    output_file = output_folder + "/"
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
                    quality.append(np.mean(result.iloc[0:29, 1]))
                    runtime.append(np.mean(result.iloc[0:29, 2]))
                    print(quality)
                    print(runtime)
                    f = plt.figure()
                    plt.plot(runtime)
                    plt.xlabel("Heuristic Weights")
                    plt.ylabel("Runtime (ms)")
                    plt.title("Varying heuristic weights " + dynamic_type)
                    plt.show()
                    f.savefig("Heuristic_X=10_Y=2_" + dynamic_type + ".pdf", bbox_inches='tight')


if __name__ == "__main__":
  plot_online_3d()

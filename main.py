import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

output_folder = "output_files"
dx = 3
dy = 3

def generate_table_agents():
  discount_factor = 0.9
  horizon = 4
  dynamic_types = ["FINITE_HORIZON", "INFINITE_HORIZON"]
  switching_cost = 50
  # algorithms = [("C_DPOP", "DPOP"),
  #               ("LS_SDPOP", "DPOP"), ("LS_SDPOP", "MGM"), ("LS_RAND", "MGM"),
  #               ("FORWARD", "DPOP"), ("FORWARD", "MGM"),
  #               ("BACKWARD", "DPOP"), ("BACKWARD", "MGM")
  #               ]
  algorithms = [("LS_SDPOP", "MGM"), ("FORWARD", "MGM"), ("BACKWARD", "MGM")]
  tables = {}
  for dynamic_type in dynamic_types:
    df = pd.DataFrame(columns=algorithms)
    for (pdcop_algorithm, dcop_algorithm) in algorithms:
      if (pdcop_algorithm, dcop_algorithm) == ("LS_SDPOP", "DPOP"):
        heuristic_weight = 0.6
      else:
        heuristic_weight = 0.0
      for decision in range(10, 30, 5):
        output_file = output_folder + "/"
        output_file += dynamic_type + "/"
        output_file += "x=" + str(decision)
        output_file += "_y=" + str(int(decision/5))
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
        # print(result_instance.columns)
        quality = np.mean(result_instance["Utility"])
        runtime = np.mean(result_instance["Time (ms)"])
        df.loc[decision, (pdcop_algorithm, dcop_algorithm)] = (quality, runtime)
    tables[dynamic_type] = df
    df.to_csv("dynamic_type" + "csv")
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
  generate_table_agents()

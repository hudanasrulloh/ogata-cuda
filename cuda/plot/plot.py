import numpy as np
import matplotlib.pyplot as plt
import subprocess

# Range of q values
q_values = np.linspace(0.01, 1.0, 100)

# Output lists
exact_results = []
numerical_transformed_results_01 = []
numerical_transformed_results_02 = []
numerical_transformed_results_03 = []
numerical_transformed_results_04 = []

# Call your CUDA program with each q value and store the results
for q in q_values:
    output = subprocess.check_output(["./example", str(q)]).decode("utf-8").strip() # suppose your exeecute file's name is example

    output_lines = output.split("\n")  # split the output string into lines
    results = {}  # store the results in a dictionary
    for line in output_lines:
        label, number_string = line.split("=")  # split each line into a label and a number string
        number = float(number_string)  # convert the number string to a float
        results[label.strip()] = number  # store the number in the dictionary under the label

    # Save the results in the lists
    exact_results.append(results["Exact"])
    numerical_transformed_results_01.append(results["Numerical Ogata Opt_t"])
    numerical_transformed_results_02.append(results["Numerical Ogata UnOpt"])
    numerical_transformed_results_03.append(results["Numerical Ogata h-fixed (0.05)"])
    numerical_transformed_results_04.append(results["Numerical Ogata SetDef_"])

# Plot the results
plt.figure(figsize=(8,4))
plt.plot(q_values, exact_results, label="Exact", linestyle='dashdot')
plt.plot(q_values, numerical_transformed_results_01, label="Numerical Ogata Opt_t")
plt.plot(q_values, numerical_transformed_results_02, label="Numerical Ogata UnOpt")
plt.plot(q_values, numerical_transformed_results_03, label="Numerical Ogata h-fixed (0.05)")
plt.plot(q_values, numerical_transformed_results_04, label="Numerical Ogata SetDef_", linestyle='--')

plt.legend()
plt.xlabel('q')
plt.ylabel('Result')
plt.title('Comparison of results')
plt.grid(True)
plt.show()

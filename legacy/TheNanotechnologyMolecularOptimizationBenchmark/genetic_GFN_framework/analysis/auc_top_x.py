"""
This script calculates the AUC (Area Under the Curve) for the top X predictions
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import auc

def calculate_auc_top_x(final_history, top_x):
    """
    Calculate the AUC for the top X fitness values at each step.

    Args:
        final_history (pd.DataFrame): DataFrame containing 'step' and 'fitness' columns.
        top_x (int): Number of top fitness values to consider for AUC calculation.

    Returns:
        float: The calculated AUC value.
        np.ndarray: Mean fitness values for each step.
        np.ndarray: Standard deviation of fitness values for each step.
        np.ndarray: Oracle call checkpoints
    """

    # Get the unique, sorted oracle call checkpoints which will serve as the x-axis
    oracle_calls = np.sort(final_history['oracle_calls'].unique())

    mean_fitness_values = []
    std_fitness_values = []
    best_fitness_values = []

    if len(oracle_calls) > 10000:
        oracle_calls = oracle_calls[:10000]

    oracle_calls_max = np.max(oracle_calls)

    sum_expected = (oracle_calls_max)*(oracle_calls_max+1)/2
    assert np.sum(oracle_calls) == sum_expected, f"The oracle calls sum should be {sum_expected} but is {np.sum(oracle_calls)}, {oracle_calls}"

    for step in oracle_calls:
        # Get all fitness values recorded up to and including the current step
        cumulative_fitness = final_history[final_history['oracle_calls'] <= step]['fitness']

        # Sort the values in descending order to find the best performers
        best_fitness_so_far = np.sort(cumulative_fitness)[::-1]

        # Select the top X fitness values. If fewer than X values exist, take all available.
        top_performers = best_fitness_so_far[:top_x]
        best_fitness_values.append(best_fitness_so_far[0])

        # Calculate the mean and standard deviation for the top performers and store them
        mean_fitness_values.append(np.mean(top_performers))
        std_fitness_values.append(np.std(top_performers))

    mean_fitness_values = np.array(mean_fitness_values)
    std_fitness_values = np.array(std_fitness_values)
    best_fitness_values = np.array(best_fitness_values)

    auc_value = auc(oracle_calls, mean_fitness_values)/len(oracle_calls)

    return auc_value, mean_fitness_values, std_fitness_values, oracle_calls




    mean_values = np.nanmean(auc_values, axis=1)
    std_values = np.nanstd(auc_values, axis=1)
    auc_x = np.trapz(mean_values, x=np.arange(0, n_calls))

    return auc_x, mean_values, std_values, oracle_checkpoints

def analyze_final_history(final_history_path, top_x, output_path):
    """
    Analyze the final history DataFrame to calculate AUC and statistics.

    Args:
        final_history_path (str): Path to the final history CSV file.
        top_x (int): Number of top fitness values to consider for AUC calculation.
        output_path (str): Path to save the output files.

    Returns:
    """
    final_history = pd.read_csv(final_history_path, sep=';')
    auc_x, mean_values, std_values, oracle_checkpoints = calculate_auc_top_x(final_history, top_x)
    _, best, _, oracle_checkpoints_best = calculate_auc_top_x(final_history, 1)
    output_file = f"{output_path}/auc_top_{top_x}.txt"
    np.savetxt(output_file, np.column_stack((oracle_checkpoints, mean_values, std_values)), header=f"AUC: {auc_x}", fmt='%.6f', delimiter='\t')

    fig, ax = plt.subplots()
    ax.plot(oracle_checkpoints, mean_values)
    ax.fill_between(oracle_checkpoints, mean_values - std_values, mean_values + std_values, alpha=0.2, label = f"<Top {top_x}>")
    ax.plot(oracle_checkpoints_best, best, linestyle='-', color='red', label=f"<Top {1}>")
    ax.set_ylabel(f'<Top x Fitness Values>')
    ax.set_title(f'AUC: {auc_x:.3f}')
    ax.set_xlabel('Step')
    ax.legend()

    output_file = f"{output_path}/auc_top_{top_x}.svg"
    plt.savefig(output_file)
    plt.close()


if __name__ == '__main__':
    final_history_path = sys.argv[1]
    top_x = int(sys.argv[2])
    output_path = sys.argv[3]

    analyze_final_history(final_history_path, top_x, output_path)







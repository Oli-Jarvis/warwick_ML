import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import Counter
import re

POP_TOKEN = '[pop]'
FRAGMENT_REGEX = r'frag_(\d+)'
TOP_N_FRAGMENTS = 20
BIN_SIZE = 500

def plot_pop_tokens(pop_counts_by_bin, output_filename="pop_token_plot.png"):
    """
    Creates and saves a bar chart of pop token counts per bin.
    """
    if pop_counts_by_bin.empty:
        print("No pop token data to plot.")
        return

    plt.figure(figsize=(12, 7))
    pop_counts_by_bin.plot(kind='bar', color='steelblue', alpha=0.9, width=0.85)

    plt.title(f"Total '{POP_TOKEN}' Token Count by Oracle Call Bin (Bin Size={BIN_SIZE})", fontsize=16, pad=20)
    plt.xlabel("Oracle Call Bin", fontsize=12)
    plt.ylabel("Total Count", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(f"{output_filename}/pop_freq.svg", bbox_inches='tight')
    plt.close()


def plot_fragment_frequencies(fragment_data_table, output_filename="fragment_frequency_plot.png"):
    """
    Creates and saves a stacked bar chart of top fragment frequencies per bin.
    """
    if fragment_data_table.empty:
        print("No fragment data to plot.")
        return


    fragment_data_table.columns = [f"frag_{c}" for c in fragment_data_table.columns]

    ax = fragment_data_table.plot(
        kind='bar',
        stacked=True,
        figsize=(14, 8),
        cmap='tab20'  # Use a colormap with good differentiation
    )

    plt.title(f"Top {TOP_N_FRAGMENTS} Fragment Frequency by Oracle Call Bin (Stacked)", fontsize=16, pad=20)
    plt.xlabel("Oracle Call Bin", fontsize=12)
    plt.ylabel("Frequency Count", fontsize=12)
    plt.xticks(rotation=45, ha='right')


    plt.legend(title='Fragments', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(f"{output_filename}/fragment_freqs.svg", bbox_inches='tight')
    plt.close()

def process_and_aggregate_data(df, output_path):

    if 'encoding' not in df.columns or 'oracle_calls' not in df.columns:
        print("Error: CSV must contain 'encoding' and 'oracle_calls' columns.")
        return

    # handle NaN values
    df['encoding'] = df['encoding'].fillna('')


    # extract pop token
    df['pop_count'] = df['encoding'].astype(str).str.count(re.escape(POP_TOKEN))

    # extract fragment lists
    df['fragments'] = df['encoding'].astype(str).str.findall(FRAGMENT_REGEX)

    # create bis for 'oracle_calls'
    max_calls = df['oracle_calls'].max()
    if pd.isna(max_calls):
        print("Error: 'oracle_calls' column contains no valid data.")
        return

    bins = list(range(-1, int(max_calls + BIN_SIZE), BIN_SIZE))
    bin_labels = [f"({bins[i]}, {bins[i + 1]}]" for i in range(len(bins) - 1)]

    if not bin_labels:
        print("Error: Could not create any bins for 'oracle_calls'.")
        return


    df['oracle_bin'] = pd.cut(df['oracle_calls'],
                              bins=bins,
                              labels=bin_labels,
                              right=True)

    # Ensure 'oracle_bin' is treated as a categorical type for correct grouping
    df['oracle_bin'] = df['oracle_bin'].astype('category')




    pop_counts_by_bin = df.groupby('oracle_bin', observed=True)['pop_count'].sum()

    # frag feqs
    # explode the 'fragments' list into separate rows
    df_exploded = df.explode('fragments').dropna(subset=['fragments'])

    if df_exploded.empty:
        print("No fragments found after parsing. Skipping fragment plot.")
        fragment_data_table = pd.DataFrame()
    else:
        # Find the top N most common fragments overall
        top_fragments = df_exploded['fragments'].value_counts().head(TOP_N_FRAGMENTS).index

        # Filter the exploded data to *only* include these top fragments
        df_top_fragments = df_exploded[df_exploded['fragments'].isin(top_fragments)]

        # Group by bin *and* fragment, then count occurrences
        fragment_counts = df_top_fragments.groupby(['oracle_bin', 'fragments'], observed=True).size()

        # unstack to create a table for plotting
        fragment_data_table = fragment_counts.unstack(fill_value=0)

    # --- Generate Plots ---
    plot_pop_tokens(pop_counts_by_bin, output_path)
    plot_fragment_frequencies(fragment_data_table, output_path)



def analyze_ga_statistics(genetic_search_rows, output_file_path=None):
    """
    Analyzes and prints crossover and mutation statistics from a genetic algorithm run.

    The report is always printed to the console.
    If output_file_path is provided, the report is also saved to that file.

    Args:
        genetic_search_rows (pd.DataFrame): A DataFrame containing columns
                                            'crossover_stats' (boolean or 0/1) and
                                            'mutation_stats' (int, -1 for no mutation).
        output_file_path (str, optional): The path to the file where the
                                          report should be saved.

    Returns:
        str: The generated statistics report as a single string.
    """
    report_lines = []

    if 'crossover_stats' not in genetic_search_rows.columns or \
       'mutation_stats' not in genetic_search_rows.columns:
        error_msg = "Error: DataFrame must contain 'crossover_stats' and 'mutation_stats' columns."
        print(error_msg, file=sys.stderr)
        return error_msg

    crossover_stats = genetic_search_rows['crossover_stats']
    mutation_stats = genetic_search_rows['mutation_stats']

    # Crossover stats
    report_lines.append("--- Genetic Algorithm Statistics ---")
    report_lines.append("\n## Crossover Stats")
    total_ops = len(crossover_stats)

    if total_ops > 0:
        crossover_count = sum(crossover_stats.astype(bool))
        crossover_percentage = (crossover_count / total_ops) * 100

        report_lines.append(f"Total operations: {total_ops}")
        report_lines.append(f"Crossovers performed: {crossover_count}")
        report_lines.append(f"Crossover Percentage: {crossover_percentage:.2f}%")
    else:
        report_lines.append("No crossover data found.")

    #Mut stats
    report_lines.append("\n## Mutation Stats")
    total_mutations = len(mutation_stats)

    if total_mutations > 0:
        mutation_counts = Counter(mutation_stats)

        report_lines.append(f"Total operations: {total_mutations}")
        report_lines.append("Mutation Type Breakdown:")

        #sort
        for mutation_type, count in mutation_counts.most_common():
            percentage = (count / total_mutations) * 100
            type_str = f"Type {mutation_type}" if mutation_type != -1 else "No Mutation (-1)"
            report_lines.append(f"  - {type_str:<20}: {percentage:>6.2f}% ({count} occurrences)")
    else:
        report_lines.append("No mutation data found.")

    #Invalid Operations -> no crossover and no mutation
    report_lines.append("\n## Invalid Operations")
    if total_ops > 0:
        invalid_subset = genetic_search_rows[
            (genetic_search_rows['crossover_stats'] == 0) &
            (genetic_search_rows['mutation_stats'] == -1)
        ]

        invalid_count = len(invalid_subset)
        invalid_percentage = (invalid_count / total_ops) * 100

        report_lines.append(f"Invalid operations (no crossover AND no mutation): {invalid_count}")
        report_lines.append(f"Percentage of total operations: {invalid_percentage:.2f}%")
    else:
        report_lines.append("No data to analyze for invalid operations.")

    #Just crossover. No mutation
    report_lines.append("\n## Just crossover")
    if total_ops > 0:
        subset = genetic_search_rows[
            (genetic_search_rows['crossover_stats'] == 1) &
            (genetic_search_rows['mutation_stats'] == -1)
        ]

        count = len(subset)
        percentage = (count / total_ops) * 100

        report_lines.append(f"Percentage of just crossover: {percentage:.2f}%")
    else:
        report_lines.append("No data to analyze for  just crossover.")

    #Just mutation. No crossover
    report_lines.append("\n## Just mutation")
    if total_ops > 0:
        subset = genetic_search_rows[
            (genetic_search_rows['crossover_stats'] == 0) &
            (genetic_search_rows['mutation_stats'] != -1)
        ]

        count = len(subset)
        percentage = (count / total_ops) * 100

        report_lines.append(f"Percentage of just mutation: {percentage:.2f}%")
    else:
        report_lines.append("No data to analyze for  just mutation.")


    final_report = "\n".join(report_lines)


    if output_file_path:
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(final_report)
        except IOError as e:
            print(f"\n--- Error: Could not save report to {output_file_path}. {e} ---", file=sys.stderr)

    return final_report

def plot_full_history(final_history_csv, output_path):
    """
    Plots the full history of fitness values against the number of oracle calls.
    Different methods (e.g., 'agent_sampling', 'genetic_search') are represented with different colors.

    Args:
        final_history_csv (pd.DataFrame): DataFrame containing 'oracle_calls', 'fitness', and 'created_by' columns.
    """


    final_history_csv = pd.read_csv(final_history_csv, sep=';')

    process_and_aggregate_data(final_history_csv, output_path)

    # take rows where "created by" is "agent_sampling"
    agent_sampling_rows = final_history_csv[final_history_csv['created_by'] == 'agent_sampling']
    genetic_search_rows = final_history_csv[final_history_csv['created_by'] == 'genetic_search']

    # create lists with n_oracle_calls and fitness values
    agent_sampling_calls = agent_sampling_rows['oracle_calls'].tolist()
    agent_sampling_fitness = agent_sampling_rows['fitness'].tolist()

    genetic_search_calls = genetic_search_rows['oracle_calls'].tolist()
    genetic_search_fitness = genetic_search_rows['fitness'].tolist()

    fig, ax = plt.subplots()

    ax.scatter(agent_sampling_calls, agent_sampling_fitness, label='Agent Sampling', color='blue', alpha=0.6)
    if "crossover_stats" in genetic_search_rows.keys():
        crossover_stats = genetic_search_rows['crossover_stats'].tolist()
    else:
        crossover_stats = [-1]
    if "mutation_stats" in genetic_search_rows.keys():
        mutation_stats = genetic_search_rows['mutation_stats'].tolist()
    else:
        mutation_stats = [-1]
    #check if all entries are "-1" -> not tracked
    if all(m == -1 for m in mutation_stats) == False and all(c == 0 for c in crossover_stats) == False:

        analyze_ga_statistics(genetic_search_rows, f"{output_path}/ga_stats.txt")


        crossover_markers = {True: 'x', False: 'o'}

        # Get unique mutation types and assign colors
        unique_mutations = sorted(genetic_search_rows['mutation_stats'].unique())
        cmap = plt.get_cmap('Set1')  # A good colormap for categories
        mutation_colors = {mut_type: cmap(i % 10) for i, mut_type in enumerate(unique_mutations)}

        existing_handles, _ = ax.get_legend_handles_labels()

        # Plot each combination
        for crossover_status, marker in crossover_markers.items():
            for mut_type, color in mutation_colors.items():

                # Filter data for this specific combination
                subset = genetic_search_rows[
                    (genetic_search_rows['crossover_stats'] == crossover_status) &
                    (genetic_search_rows['mutation_stats'] == mut_type)
                    ]

                if not subset.empty:
                    label = f'Mut_Type {mut_type} (Crossover: {crossover_status})'
                    ax.scatter(
                        subset['oracle_calls'],
                        subset['fitness'],
                        color=color,
                        marker=marker,
                        alpha=0.7,
                        s=50  # Make markers a bit larger
                    )
        color_handles = [
            Line2D([0], [0], marker='o', color='w',
                   label=f'Mut_Type {mut_type}',
                   markerfacecolor=color, markersize=10)
            for mut_type, color in mutation_colors.items()
        ]
        marker_handles = [
            Line2D([0], [0], marker=marker, color='black', linestyle='None',
                   label=f'Crossover: {status}', markersize=10)
            for status, marker in crossover_markers.items()
        ]
        ax.legend(
            handles=existing_handles + color_handles + marker_handles,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.
        )
    else:
        ax.scatter(genetic_search_calls, genetic_search_fitness, label='Genetic Search', color='orange', alpha=0.6)
        ax.legend()

    ax.set_xlabel('Number of Oracle Calls')
    ax.set_ylabel('Fitness')


    plt.savefig(f"{output_path}/history.svg", bbox_inches='tight')
    plt.clf()
    plt.close()

if __name__ == '__main__':

    final_history_csv = sys.argv[1]
    output_path = sys.argv[2]
    plot_full_history(final_history_csv, output_path)



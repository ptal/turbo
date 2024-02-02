import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from packaging import version

# A tentative to have unique experiment names.
def make_uid(config, version, machine, eps_num_subproblems, or_nodes, and_nodes):
  if config == 'TurboGPU':
    return 'TurboGPU_' + str(version) + '_' + machine + '_' + str(eps_num_subproblems) + '_' + str(or_nodes) + '_' + str(and_nodes)
  elif config == 'TurboCPU':
    return 'TurboCPU_' + str(version) + '_' + machine
  else:
    return config + '_' + str(version) + '_' + machine

# For some experiments, I put too much information in the configuration name, so we clean it up here.
def clean_config_name(config):
  if 'TurboGPU' in config:
    return 'TurboGPU'
  elif 'TurboCPU' in config:
    return 'TurboCPU'
  else:
    return config

def read_experiments(experiments):
  all_xp = pd.read_csv("../campaign/baseline.csv")
  all_xp['hardware'] = 'Intel Core i9-10900X@3.7GHz;24GO DDR4;NVIDIA RTX A5000'
  all_xp['version'] = all_xp['configuration'].apply(determine_version)
  for e in experiments:
    df = pd.read_csv(e)
    df['configuration'] = df['configuration'].apply(clean_config_name)
    df = df[(df['configuration'] != "TurboGPU") | (~df['or_nodes'].isna())]
    df = df[(df['configuration'] != "TurboGPU") | (~df['and_nodes'].isna())]
    # all_xp = pd.merge(df.T, all_xp.T, left_index=True, right_index=True, how='outer').T
    all_xp = pd.concat([df, all_xp], ignore_index=True)
  all_xp['version'] = all_xp['version'].apply(version.parse)
  all_xp['nodes'] = all_xp['nodes'].fillna(0).astype(int)
  all_xp['or_nodes'] = all_xp['or_nodes'].fillna(1).astype(int)
  all_xp['and_nodes'] = all_xp['and_nodes'].fillna(1).astype(int)
  all_xp['fixpoint_iterations'] = pd.to_numeric(all_xp['fixpoint_iterations'], errors='coerce').fillna(0).astype(int)
  all_xp['eps_num_subproblems'] = pd.to_numeric(all_xp['eps_num_subproblems'], errors='coerce').fillna(1).astype(int)
  all_xp['machine'] = all_xp['hardware'].apply(determine_machine)
  all_xp['uid'] = all_xp.apply(lambda row: make_uid(row['configuration'], row['version'], row['machine'],
                                                    row['eps_num_subproblems'], row['or_nodes'], row['and_nodes']), axis=1)
  all_xp['nodes_per_second'] = all_xp['nodes'] / all_xp['solveTime']
  all_xp['fp_iterations_per_node'] = all_xp['fixpoint_iterations'] / all_xp['nodes']
  return all_xp

def plot_overall_result(df):
  grouped = df.groupby(['uid', 'status']).size().unstack(fill_value=0)
  grouped['OPTIMAL/UNSAT'] = grouped.get('OPTIMAL_SOLUTION', 0) + grouped.get('UNSATISFIABLE', 0)

  grouped = grouped[['OPTIMAL/UNSAT', 'SATISFIED', 'UNKNOWN']]

  # Sort the DataFrame by 'OPTIMAL/UNSAT' and then 'SATISFIED'
  grouped.sort_values(by=['OPTIMAL/UNSAT', 'SATISFIED'], ascending=[True, True], inplace=True)

  # Plot
  colors = {'OPTIMAL/UNSAT': 'green', 'SATISFIED': 'lightblue', 'UNKNOWN': 'orange'}
  ax = grouped.plot(kind='barh', stacked=True, color=[colors[col] for col in grouped.columns])
  plt.title('Problem Status by Configuration')
  plt.ylabel('Configuration')
  plt.xlabel('Number of Problems')
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
  plt.tight_layout()
  plt.show()

def remove_meluxina_scaling_tests(df):
  return df[(df['configuration'] != 'TurboGPU') | (df['machine'] != 'Meluxina') | (((df['eps_num_subproblems'] == 1024) | (df['eps_num_subproblems'] == 4096)) & (df['and_nodes'] == 256) & (df['or_nodes'] == 108))]

def determine_machine(hardware_info):
  if hardware_info == 'Intel Core i9-10900X@3.7GHz;24GO DDR4;NVIDIA RTX A5000':
    return 'Desktop-A5000'
  elif hardware_info == 'AMD EPYC 7452 32-Core@2.35GHz; RAM 512GO;NVIDIA A100 40GB HBM':
    return 'Meluxina'
  else:
    return 'unknown'

def determine_version(solver_info):
  if 'choco' in solver_info:
    return '4.10.13'
  elif 'or-tools' in solver_info:
    return '9.6'
  else:
    return 'unknown'

def metrics_table(df):
  grouped = df.groupby(['uid'])

  # Calculate metrics
  metrics = grouped.agg(
    version=('version', 'first'),
    machine=('machine', 'first'),
    avg_nodes_per_second=('nodes_per_second', 'mean'),
    median_nodes_per_second=('nodes_per_second', 'median'),
    avg_fp_iterations=('fp_iterations_per_node', 'mean'),
    median_fp_iterations=('fp_iterations_per_node', 'median'),
    avg_propagator_mem_mb=('propagator_mem', lambda x: x.mean() / 1000000),
    median_propagator_mem_mb=('propagator_mem', lambda x: x.median() / 1000000),
    avg_store_mem_kb=('store_mem', lambda x: x.mean() / 1000),
    median_store_mem_kb=('store_mem', lambda x: x.median() / 1000)
  )

  # Count problems with non-zero num_blocks_done and not solved to optimality or proven unsatisfiable
  condition = (df['num_blocks_done'] != 0) & (~df['status'].isin(['OPTIMAL_SOLUTION', 'UNSATISFIABLE']))
  idle_eps_workers = df[condition].groupby(['uid']).size().reset_index(name='idle_eps_workers')

  # Merge metrics with idle_eps_workers
  overall_metrics = metrics.merge(idle_eps_workers, on=['uid'], how='left').fillna(0)
  overall_metrics = overall_metrics.sort_values(by=['version', 'machine', 'avg_nodes_per_second'], ascending=[False, True, False])

  return overall_metrics

def compare_solvers_pie_chart(df, uid1, uid2):
    """
    Compares the performance of two solvers based on objective value and optimality.

    Parameters:
    - df: DataFrame containing the data.
    - uid1: Name of the first solver (str).
    - uid2: Name of the second solver (str).

    Returns:
    - Displays a pie chart comparing the performance of the two solvers.
    """

    solvers_df = df[(df['uid'] == uid1) | (df['uid'] == uid2)]

    # Pivoting for 'objective', 'method', and 'status' columns
    pivot_df = solvers_df.pivot_table(index='data_file', columns='uid', values=['objective', 'method', 'status'], aggfunc='first')

    # Compare objective values based on method and optimality status
    conditions = [
        # Error
        (pivot_df['method', uid1] != pivot_df['method', uid2]),

        # Solver 1 better
        ((pivot_df['status', uid1] != "UNKNOWN") & (pivot_df['status', uid2] == "UNKNOWN")) |
        ((pivot_df['method', uid1] == "minimize") & (pivot_df['objective', uid1] < pivot_df['objective', uid2])) |
        ((pivot_df['method', uid1] == "maximize") & (pivot_df['objective', uid1] > pivot_df['objective', uid2])) |
        ((pivot_df['objective', uid1] == pivot_df['objective', uid2]) & (pivot_df['status', uid1] == "OPTIMAL_SOLUTION") & (pivot_df['status', uid2] != "OPTIMAL_SOLUTION")),

        # Solver 2 better
        ((pivot_df['status', uid1] == "UNKNOWN") & (pivot_df['status', uid2] != "UNKNOWN")) |
        ((pivot_df['method', uid1] == "minimize") & (pivot_df['objective', uid1] > pivot_df['objective', uid2])) |
        ((pivot_df['method', uid1] == "maximize") & (pivot_df['objective', uid1] < pivot_df['objective', uid2])) |
        ((pivot_df['objective', uid1] == pivot_df['objective', uid2]) & (pivot_df['status', uid1] != "OPTIMAL_SOLUTION") & (pivot_df['status', uid2] == "OPTIMAL_SOLUTION")),

        # Equal
        (pivot_df['status', uid1] == pivot_df['status', uid2])
    ]

    choices = ['Error', f'{uid1} better', f'{uid2} better', 'Equal']

    pivot_df['Comparison'] = np.select(conditions, choices, default='Unknown')

    # Get problems with "Unknown" comparison (should not happen, this is for debugging).
    # unknown_problems = pivot_df[pivot_df['Comparison'] == 'Unknown'].index.tolist()
    # if unknown_problems:
    #     print(f"The comparison is 'Unknown' for the following problems: {', '.join(unknown_problems)}")
    # else:
    #     print("There are no problems with 'Unknown' comparison.")

    # Get counts for each category
    category_counts = pivot_df['Comparison'].value_counts()

    color_mapping = {
        f'{uid1} better': 'green' if category_counts.get(f'{uid1} better', 0) >= category_counts.get(f'{uid2} better', 0) else 'orange',
        f'{uid2} better': 'green' if category_counts.get(f'{uid2} better', 0) > category_counts.get(f'{uid1} better', 0) else 'orange',
        'Equal': (0.678, 0.847, 0.902), # light blue
        'Unknown': 'red',
        'Error': 'red'
    }
    colors = [color_mapping[cat] for cat in category_counts.index]

    # Plot pie chart
    plt.figure(figsize=(10, 6))
    category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title(f'Objective Value and Optimality Comparison between {uid1} and {uid2}')
    plt.ylabel('')
    plt.show()

    return pivot_df

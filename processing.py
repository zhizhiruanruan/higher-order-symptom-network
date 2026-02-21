import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations


def normalize_data(data):
    col_means = data.iloc[:, 1:].mean()
    data_normalized = data.iloc[:, 1:].div(col_means)
    data_normalized.insert(0, 'ID', data['ID'])
    return data_normalized

def generate_new_variables(data):
    '''Production Interaction Item'''
    original_columns = data.columns[1:]
    
    data[original_columns] = data[original_columns].apply(pd.to_numeric)
    
    new_columns = {}
    for col1, col2 in combinations(original_columns, 2):
        new_var_name = f"{col1}{col2}"
        new_columns[new_var_name] = data[col1] * data[col2]
    
    new_data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)
    
    return new_data

def calculate_spearman_correlation(data):
    data_without_id = data.iloc[:, 1:]  
    columns = data_without_id.columns
    num_columns = len(columns)
    
    cor_matrix = np.zeros((num_columns, num_columns))
    p_matrix = np.zeros((num_columns, num_columns))
    
    for i in range(num_columns):
        for j in range(i, num_columns):
            if i == j:
                cor_matrix[i, j] = 1.0 
                p_matrix[i, j] = 0.0    
            else:
                cor, p = spearmanr(data_without_id.iloc[:, i], data_without_id.iloc[:, j])
                cor_matrix[i, j] = cor
                cor_matrix[j, i] = cor  
                p_matrix[i, j] = p
                p_matrix[j, i] = p  
    

    cor_matrix_df = pd.DataFrame(cor_matrix, columns=columns, index=columns)
    p_matrix_df = pd.DataFrame(p_matrix, columns=columns, index=columns)
    
    significant_cor_matrix_df = cor_matrix_df.copy()
    significant_cor_matrix_df[p_matrix_df >= 0.05] = 0
    
    return significant_cor_matrix_df

def extract_nonzero_correlations(matrix):
    variable_names = matrix.columns.tolist()
    results = []
    num_variables = matrix.shape[0]
    for i in range(num_variables):
        for j in range(i+1, num_variables): 
            if matrix.iloc[i, j] != 0:
                results.append({'Node1': variable_names[i], 'Node2': variable_names[j], 'Correlation': matrix.iloc[i, j]})
    return pd.DataFrame(results)


def filter_duplicate_nodes(data):
    def has_duplicate_letters(node1, node2):
        combined = node1 + node2
        return len(set(combined)) != len(combined)
    
    return data[~data.apply(lambda x: has_duplicate_letters(x['Node1'], x['Node2']) or len(x['Node1'] + x['Node2']) <= 2, axis=1)]


def process_edges(data):
    '''Separate nodes'''
    edge_list = {}
    for i, row in data.iterrows():
        nodes_split = sorted(set(row['Node1'] + row['Node2']))
        nodes_str = ','.join(nodes_split)
        if nodes_str not in edge_list:
            edge_list[nodes_str] = [row['Correlation']]
        else:
            edge_list[nodes_str].append(row['Correlation'])

    result = []
    for edge, correlations in edge_list.items():
        result.append({
            'edge': edge,
            'correlation_mean': np.mean(correlations)
        })

    result_df = pd.DataFrame(result)
    
    result_df = result_df.sort_values(by='correlation_mean', ascending=False)
    
    result_df['edge_id'] = range(1, len(result_df) + 1)

    result_top50 = result_df.head(50)

    edges = [edge.split(',') for edge in result_top50['edge']]
    edge_weights = result_top50['correlation_mean'].tolist()

    result_top50_df = pd.DataFrame({
        'edge': [f'edge{idx+1}' for idx in range(len(edges))],
        'nodes': edges, 
        'cor': edge_weights
    })
    formatted_result = [
        f"edge {row['edge_id']}: {row['edge']} cor: {row['correlation_mean']:.7f}"
        for _, row in result_top50.iterrows()
    ]
    return  result_top50_df,formatted_result
    


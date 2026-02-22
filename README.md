# Identifying key psychological symptoms by a higher-order network-based approach

This repository contains data and code for constructing and visualizing higher-order symptom networks using hypergraph analysis.

## 📁 Repository Structure


## 📊 Datasets

The repository includes three independent datasets:

| Dataset | Description | File |
|---------|-------------|------|
| **CHARLS** | China Health and Retirement Longitudinal Study | `Data/CHARLS_data_all` |
| **NHANES** | National Health and Nutrition Examination Survey | `Data/NHANES_data_all` |
| **NHRVS** | A nationally representative sample of U.S. veterans | `Data/NHRVS_data_all` |

**Data Format**: Each dataset should contain:
- First column: `ID` (participant identifier)
- Subsequent columns: Symptom/item scores (numeric)

## 🔧 Installation
```bash

pip install pandas numpy scipy xgi matplotlib
```

🚀 
Step 1: Data Processing
```python
Import the processing module and run the analysis pipeline:
import pandas as pd
import processing as dp

# Load dataset
data = pd.read_excel('Data/CHARLS_data_all')

# Step 1: Normalize data (divide by column means)
normalized_data = dp.normalize_data(data)

# Step 2: Generate interaction terms (symptom pairs)
new_data = dp.generate_new_variables(normalized_data)

# Step 3: Calculate Spearman correlations (p < 0.05)
correlation_matrix = dp.calculate_spearman_correlation(new_data)

# Step 4: Extract significant correlations
nonzero_correlations = dp.extract_nonzero_correlations(correlation_matrix)

# Step 5: Filter redundant edges
filtered_data = dp.filter_duplicate_nodes(nonzero_correlations)

# Step 6: Process top 50 edges
result_df, formatted_output = dp.process_edges(filtered_data)
```
🚀 
Step 2: Network Visualization
```python
Open Figure.ipynb to visualize the hypergraph network:
import xgi
import matplotlib.pyplot as plt

# Load processed edges
hyperedges = {}  # Load from processed data

# Create hypergraph
H = xgi.Hypergraph(hyperedges)

# Draw network
pos = xgi.barycenter_spring_layout(H)
ax, collections = xgi.draw(
    H,
    pos=pos,
    node_fc=[H.degree(node) for node in H.nodes],
    edge_fc=[H.size(edge) for edge in H.edges],
    node_size=25
)
plt.show()
```
📤 Output Format
The process_edges() function returns:
result_df: DataFrame with columns ['edge', 'nodes', 'cor']
formatted_output: List of formatted strings for easy reading



#%%
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.multitest as ssm
import scipy as sp
import pickle
import sys
import os
import re
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import plotly.express as px
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests


sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# %%
import pronto
ont = pronto.Ontology( 'http://purl.obolibrary.org/obo/go/go-basic.obo')
root = ont['GO:0008150']

#%%
# Function to get children up to a certain depth
def get_upper_terms(term, max_depth, current_depth=0):
    upper_terms = []
    if current_depth < max_depth:
        # Check direct children only
        for child in term.subclasses(distance=1):  # Use subclasses with distance=1 for direct children
            upper_terms.append((child.id, child.name))
            # Recursive call to explore further down to specified depth
            upper_terms.extend(get_upper_terms(child, max_depth, current_depth + 1))
    return set(upper_terms)

# Example: Get terms up to level 2
# Assuming 'GO:0008150' is the biological process root term, modify as necessary for your term
terms = get_upper_terms(root, 5) - get_upper_terms(root, 4)
print(terms)


# %%
import random
random_terms = random.sample(terms, 30)
go_term_ids = [term[0] for term in random_terms]


# %%
def read_go_file(file_path):
    go_terms = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into GO term and genes, assuming they are separated by tabs
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                go_term, genes = parts
                genes_list = genes.split()  # Splitting gene symbols which are separated by spaces
                go_terms[go_term] = genes_list
    return go_terms

go_terms = read_go_file('/home/jiye/jiye/copycomparison/OC_transcriptome/GO_Biological_Process_2021.txt')

targetgenelist = []
go_term_ids=['GO:0006302']

for term in go_term_ids:
    for key, genes in go_terms.items():
            if term in key:
                targetgenelist.extend(genes)
# %%

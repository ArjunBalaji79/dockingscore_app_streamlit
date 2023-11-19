import streamlit as st
from utils import load_model, smiles_to_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem
import os
import numpy as np

# Preset data for dropdown menus
organs = {
    'Brain': ['O14672', 'P07900', 'P35869', 'P40763', 'P49841', 'Q9UBS5', 'Q00535', 'Q11130', 'Q16539', 'P05129'], 
    'Organ2': ['Protein3', 'Protein4']
}
models = ['GCN', 'GCN+GAT']

def main():
    st.title('Docking Score Prediction')

    # Organ Selection
    selected_organ = st.selectbox('Select Organ', list(organs.keys()))

    # Protein Selection
    proteins = organs[selected_organ]
    selected_protein = st.selectbox('Select Protein', proteins)

    # Model Selection
    selected_model = st.selectbox('Select Model', models)

    # SMILES Input
    smiles_string = st.text_input('Enter SMILES String')

    # Predict Button
    if st.button('Predict'):
        try:
            # Load the model
            model = load_model(selected_model, selected_protein)
            
            # Convert SMILES to graph
            graph = smiles_to_graph(smiles_string)

            if graph is not None:
                # Make prediction
                model.eval()
                with torch.no_grad():
                    prediction = model(graph)
                    predicted_score = prediction.item()
                st.write(f'Predicted Docking Score: {predicted_score}')
            else:
                st.error('Invalid SMILES string')
        except Exception as e:
            st.error(f'An error occurred: {e}')

if __name__ == '__main__':
    main()

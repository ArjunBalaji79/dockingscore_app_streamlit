import streamlit as st
from utils import load_gcn_model, load_gcn_gat_model, smiles_to_graph
import torch
import numpy as np
from PIL import Image
import os
import stmol
from stmol import *
from stmol import showmol
import py3Dmol
import csv

# Function to generate dictionary mapping current names to new names from CSV
def generate_name_mapping(csv_file_path):
    name_mapping = {}
    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header row if exists
        for row in csv_reader:
            uniprot_id = row[1]
            protein_name = row[2]
            name_mapping[uniprot_id] = protein_name
    return name_mapping

# Preset data for dropdown menus
organs = {
    'Brain': ['O14672', 'P07900', 'P35869', 'P40763', 'P49841', 'Q9UBS5l', 'Q00535', 'Q11130', 'Q16539', 'P05129'], 
    'Liver': ['P04150', 'P14555', 'P19793', 'P07900_Liver', 'P22845', 'P42574', 'P55210', 'Q15465', 'P35869_Liver', 'Q96RI1'],
    'Kidney': ['O14920', 'P12821', 'P35869_Kidney', 'P42574_Kidney', 'P55210_Kidney', 'Q15303', 'Q16236', 'Q16665', 'P41595','P80365']
}

def main():
    st.title('ProteoDockNet: A Graph Neural Network Based Platform for Docking Score Prediction')

    # Sidebar for user input
    st.sidebar.header("User Input Features")
    
    # Path to the CSV file containing name mapping
    csv_file_path = 'Protein-list - Sheet1.csv'
    
    # Generate name mapping dictionary
    name_mapping = generate_name_mapping(csv_file_path)
    
    # Update organs dictionary using name mapping
    updated_organs = {}
    for organ, proteins in organs.items():
        updated_proteins = []
        for protein in proteins:
            updated_proteins.append(name_mapping.get(protein, protein))
        updated_organs[organ] = updated_proteins

    selected_organ = st.sidebar.selectbox('Select Organ', list(updated_organs.keys()))
    proteins = updated_organs[selected_organ]

    selected_protein_display = st.sidebar.selectbox('Select Protein', proteins)
    selected_protein = selected_protein_display  # Keep a copy of the displayed protein name

# Convert the displayed protein name back to the original UniProt name
    for uni_id, name in name_mapping.items():
        if name == selected_protein_display:
            selected_protein = uni_id
            break

    selected_model = st.sidebar.selectbox('Select Model', ['GCN', 'GCN+GAT'])

    # Main panel
    st.write("## Prediction Results")
    # model_image = Image.open('download-2.png')
    # st.image(model_image, caption='Model Architecture',use_column_width=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Using markdown with HTML to style the button 
        smiles_string = st.text_input('Enter SMILES String', key='smiles_input')
        st.markdown(
            """
            <style>
                div.stButton > button:first-child {
                    width: 100%;
                    height: 50px;  # Custom height
                    font-size: 20px;  # Larger font size
                }
            </style>""",
            unsafe_allow_html=True,
        )
    
    if st.button('Predict'):
        try:
            # Load the model based on the selected_model
            if selected_model == 'GCN':
                model = load_gcn_model(selected_protein)
            elif selected_model == 'GCN+GAT':
                model = load_gcn_gat_model(selected_protein)
            else:
                raise ValueError(f"Invalid model selection: {selected_model}")

            graph = smiles_to_graph(smiles_string)  # Convert SMILES to graph

            if graph is not None:
                # Make prediction
                model.eval()
                with torch.no_grad():
                    prediction = model(graph)
                    predicted_score = prediction.item()
                    formatted_score = "{:.4f} KCal".format(predicted_score)
                st.success(f'Predicted Docking Score: {formatted_score}')
            else:
                st.error('Invalid SMILES string')
        except Exception as e:
            st.error(f'An error occurred: {e}')

    # 1A2C
    # Structure of thrombin inhibited by AERUGINOSIN298-A from a BLUE-GREEN ALGA
    xyzview = py3Dmol.view(query='pdb:1A2C') 
    xyzview.setStyle({'cartoon':{'color':'spectrum'}})
    showmol(xyzview, height = 500,width=800)
    

if __name__ == '__main__':
    main()

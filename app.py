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

# Function to convert UniProt ID to PDB ID
def uniprot_to_pdb(uniprot_id):
    try:
        # Send a request to the UniProt API to retrieve PDB mappings
        response = requests.get(f'https://www.uniprot.org/uniprot/{uniprot_id}.xml')
        xml_tree = ET.parse(io.StringIO(response.text))
        root = xml_tree.getroot()
        namespaces = {'ns': 'http://uniprot.org/uniprot'}
        
        # Find PDB ID(s) in the XML response
        pdb_ids = []
        for entry in root.findall('.//ns:dbReference[@type="PDB"]', namespaces):
            pdb_id = entry.attrib['id']
            pdb_ids.append(pdb_id)
        
        return pdb_ids
    except Exception as e:
        st.error(f'Error retrieving PDB mappings for UniProt ID {uniprot_id}: {e}')
        return []

# Function to visualize protein structure using py3Dmol
def visualize_protein(uniprot_id):
    pdb_ids = uniprot_to_pdb(uniprot_id)
    if pdb_ids:
        for pdb_id in pdb_ids:
            st.write(f'### Structure for PDB ID: {pdb_id}')
            viewer = py3Dmol.view(query=f'pdb:{pdb_id}')
            viewer.setStyle({'cartoon':{'color':'spectrum'}})
            viewer.show()
    else:
        st.warning('No PDB mappings found for the given UniProt ID.')

# Function to generate dictionary mapping current names to new names from CSV
def generate_name_mapping(csv_file_path):
    name_mapping = {}
    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header row if exists
        for row in csv_reader:
            current_name = row[1]
            new_name = row[2]
            name_mapping[current_name] = new_name
    return name_mapping

# Preset data for dropdown menus
organs = {
    'Brain': ['O14672', 'P07900', 'P35869', 'P40763', 'P49841', 'Q9UBS5l', 'Q00535', 'Q11130', 'Q16539', 'P05129'], 
    'Liver': ['P04150', 'P14555', 'P19793', 'P07900_Liver', 'P22845', 'P42574', 'P55210', 'Q15465', 'P35869_Liver', 'Q96RI1'],
    'Kidney': ['O14920', 'P12821', 'P35869_Kidney', 'P42574_Kidney', 'P55210_Kidney', 'Q15303', 'Q16236', 'Q16665', 'P41595','P80365']
}
models = ['GCN', 'GCN+GAT']

def main():
    st.title('ProteoDockNet: A Graph Neural Network Based Platform for Docking Score Prediction')

    # Sidebar for user input
    st.sidebar.header("User Input Features")
    selected_organ = st.sidebar.selectbox('Select Organ', list(organs.keys()))
    proteins = organs[selected_organ]
    selected_protein = st.sidebar.selectbox('Select Protein', proteins)
    selected_model = st.sidebar.selectbox('Select Model', models)

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

    # Visualize protein structure for the selected protein
    st.write("## Protein Structure Visualization")
    selected_uniprot_id = name_mapping.get(selected_protein, selected_protein)
    visualize_protein(selected_uniprot_id)

if __name__ == '__main__':
    # Path to the CSV file containing name mapping
    csv_file_path = 'Protein-list - Sheet1.csv'
    # Generate name mapping dictionary
    name_mapping = generate_name_mapping(csv_file_path)
    # Update organs dictionary using name mapping
    for organ, proteins in organs.items():
        updated_proteins = []
        for protein in proteins:
            updated_proteins.append(name_mapping.get(protein, protein))
        organs[organ] = updated_proteins
    main()

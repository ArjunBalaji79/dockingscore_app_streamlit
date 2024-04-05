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

# Preset data for dropdown menus
organs_with_names = {
    'Brain': {
        'O14672': 'Disintegrin and metalloproteinase domain-containing protein 10',
        'P07900': 'Heat shock protein HSP 90-alpha',
        'P35869': 'Aryl hydrocarbon receptor',
        'P40763': 'Signal transducer and activator of transcription 3',
        'P49841': 'Glycogen synthase kinase-3 beta',
        'Q9UBS5l': 'Q9UBS5l',
        'Q00535': 'Cyclin-dependent kinase 5',
        'Q11130': 'Alpha-(1,3)-fucosyltransferase 7',
        'Q16539': 'Mitogen-activated protein kinase 14',
        'P05129': 'Protein kinase C gamma type'
    },
    'Liver': {
        'P04150': 'Glucocorticoid receptor',
        'P14555': 'Phospholipase A2, membrane associated',
        'P19793': 'P19793',
        'P07900_Liver': 'P07900_Liver',
        'P22845': 'P22845',
        'P42574': 'P42574',
        'P55210': 'P55210',
        'Q15465': 'Sonic hedgehog protein',
        'P35869_Liver': 'P35869_Liver',
        'Q96RI1': 'Bile acid receptor'
    },
    'Kidney': {
        'O14920': 'Inhibitor of nuclear factor kappa-B kinase subunit beta',
        'P12821': 'P12821',
        'P35869_Kidney': 'P35869_Kidney',
        'P42574_Kidney': 'P42574_Kidney',
        'P55210_Kidney': 'P55210_Kidney',
        'Q15303': 'Q15303',
        'Q16236': 'Nuclear factor erythroid 2-related factor 2',
        'Q16665': 'Hypoxia-inducible factor 1-alpha',
        'P41595': '5-hydroxytryptamine receptor 2B',
        'P80365': '11-beta-hydroxysteroid dehydrogenase type 2'
    }
}

def main():
    st.title('ProteoDockNet: A Graph Neural Network Based Platform for Docking Score Prediction')

    # Sidebar for user input
    st.sidebar.header("User Input Features")
    
    selected_organ = st.sidebar.selectbox('Select Organ', list(organs_with_names.keys()))
    
    # Display protein names from organs_with_names dictionary in dropdown menu
    proteins_with_names = organs_with_names[selected_organ]
    selected_protein_display = st.sidebar.selectbox('Select Protein', list(proteins_with_names.values()))

    # Convert the displayed protein name back to the original UniProt ID
    selected_protein = ''
    for uni_id, name in proteins_with_names.items():
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
                
                # Display original UniProt ID used for processing
                st.write(f"Original UniProt ID: {selected_protein}")

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

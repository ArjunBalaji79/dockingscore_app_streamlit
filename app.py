import streamlit as st
from utils import load_model, smiles_to_graph
import torch
import numpy as np
from PIL import Image
import os
import stmol
from stmol import *
from stmol import showmol
import py3Dmol

# Preset data for dropdown menus
organs = {
    'Brain': ['O14672', 'P07900', 'P35869', 'P40763', 'P49841', 'Q9UBS5l', 'Q00535', 'Q11130', 'Q16539', 'P05129'], 
    'Liver': ['P04150', 'P14555', 'P19793', 'P07900_Liver', 'P22845', 'P42574', 'P55210', 'Q15465'],
    'Kidney': ['O14920', 'Protein4']
}
models = ['GCN', 'GCN+GAT']

def main():
    st.title('Graph Neural Network based Docking Score Prediction')

    # Sidebar for user input
    st.sidebar.header("User Input Features")
    selected_organ = st.sidebar.selectbox('Select Organ', list(organs.keys()))
    proteins = organs[selected_organ]
    selected_protein = st.sidebar.selectbox('Select Protein', proteins)
    selected_model = st.sidebar.selectbox('Select Model', models)

    # Main panel
    st.write("## Prediction Results and Model Architecture")
    # model_image = Image.open('download-2.png')
    # st.image(model_image, caption='Model Architecture',use_column_width=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Using markdown with HTML to style the button (limited styling)
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
            # Load the model
            model = load_model(selected_model, selected_protein)
            graph = smiles_to_graph(smiles_string)  # Convert SMILES to graph

            if graph is not None:
                # Make prediction
                model.eval()
                with torch.no_grad():
                    prediction = model(graph)
                    predicted_score = prediction.item()
                st.success(f'Predicted Docking Score: {predicted_score}')
            else:
                st.error('Invalid SMILES string')
        except Exception as e:
            st.error(f'An error occurred: {e}')

    if(selected_model=="GCN"):
      model_image = Image.open('GCNmodelflowchart.png')
      st.image(model_image, caption='GCN Model Architecture',width=200)
    else:
      model_image = Image.open('EnhancedGCNmodelflowchart.png')
      st.image(model_image, caption='GCN+GAT Model Architecture', width=250)
      



    # Optional: Additional UI elements or animations
    st.write("## Additional Information")
    st.markdown("Page is Under Construction :construction: :rotating_light: :helicopter:")
    

    # 1A2C
    # Structure of thrombin inhibited by AERUGINOSIN298-A from a BLUE-GREEN ALGA
    xyzview = py3Dmol.view(query='pdb:1A2C') 
    xyzview.setStyle({'cartoon':{'color':'spectrum'}})
    showmol(xyzview, height = 500,width=800)
    

if __name__ == '__main__':
    main()

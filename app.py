import streamlit as st
from utils import load_model, smiles_to_graph
import torch
import numpy as np
from PIL import Image
import os

# Preset data for dropdown menus
organs = {
    'Brain': ['O14672', 'P07900', 'P35869', 'P40763', 'P49841', 'Q9UBS5', 'Q00535', 'Q11130', 'Q16539', 'P05129'], 
    'Organ2': ['Protein3', 'Protein4']
}
models = ['GCN', 'GCN+GAT']

def main():
    st.title('Docking Score Prediction')

    # Sidebar for user input
    st.sidebar.header("User Input Features")
    selected_organ = st.sidebar.selectbox('Select Organ', list(organs.keys()))
    proteins = organs[selected_organ]
    selected_protein = st.sidebar.selectbox('Select Protein', proteins)
    selected_model = st.sidebar.selectbox('Select Model', models)
    smiles_string = st.sidebar.text_input('Enter SMILES String')

    # Main panel
    st.write("## Prediction Results")
    # model_image = Image.open('download-2.png')
    # st.image(model_image, caption='Model Architecture', use_column_width=True)

    if(selected_model=="GCN"):
      model_image = Image.open('GCNmodelflowchart.png')
      st.image(model_image, caption='GCN Model Architecture', width=400,height=800)
    else:
      model_image = Image.open('EnhancedGCNmodelflowchart.png')
      st.image(model_image, caption='GCN+GAT Model Architecture', width=400,height=800)
      

    # Predict Button
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

    # Optional: Additional UI elements or animations
    st.write("## Additional Information")
    st.markdown("Here you can add more information about the project, model, or any other relevant details.")

if __name__ == '__main__':
    main()

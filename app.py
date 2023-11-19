import streamlit as st
from utils import load_model, smiles_to_graph

# Preset data
organs = {'Organ1': ['Protein1', 'Protein2'], 'Organ2': ['Protein3', 'Protein4']}
models = ['gnn_model_2convlayer_5fold_withmorefeatures.pth', 'GAT']

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
        model = load_model(selected_model, selected_protein)  # Adjust based on how you load models
        graph = smiles_to_graph(smiles_string)  # Convert SMILES to graph
        if graph:
            prediction = model.predict(graph)  # Adjust based on your model's prediction method
            st.write(f'Predicted Docking Score: {prediction}')
        else:
            st.error('Invalid SMILES string')

if __name__ == '__main__':
    main()

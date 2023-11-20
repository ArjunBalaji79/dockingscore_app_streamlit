import streamlit as st
import streamlit.components.v1 as components
from utils import load_model, smiles_to_graph

# Preset data for dropdown menus
organs = {
    'Brain': ['O14672', 'P07900', 'P35869', 'P40763', 'P49841', 'Q9UBS5', 'Q00535', 'Q11130', 'Q16539', 'P05129'], 
    'Organ2': ['Protein3', 'Protein4']
}
models = ['GCN', 'GCN+GAT']

def display_mermaid_flowchart(selected_model):
    if selected_model == 'GCN':
        mermaid_script = """
        graph LR
            A[Input Data] -->|x, edge_index, batch| B(GCNConv Layer 1)
            B -->|F.relu| C(GCNConv Layer 2)
            C -->|F.relu| D[Global Mean Pooling]
            D -->|F.relu| E(Linear Layer 1)
            E --> F(Linear Layer 2 - Output)
        """
    elif selected_model == 'GCN+GAT':
        mermaid_script = """
        graph LR
            A[Input Data] -->|x, edge_index, batch| B(GCNConv Layer 1)
            B -->|BatchNorm, F.relu| C(GATConv Layer)
            C -->|BatchNorm, F.relu, Dropout| D(GCNConv Layer 2)
            D -->|BatchNorm, F.relu| E[Global Mean Pooling]
            E -->|F.relu, Dropout| F(Linear Layer 1)
            F --> G(Linear Layer 2 - Output)
        """
    
    components.html("""
    <script src="https://cdn.jsdelivr.net/npm/mermaid@8.0.0/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({startOnLoad:true});</script>
    """ + mermaid_script, height=400)

def main():
    st.title('Docking Score Prediction')

    # Sidebar for user input
    st.sidebar.header("User Input Features")
    selected_organ = st.sidebar.selectbox('Select Organ', list(organs.keys()))
    proteins = organs[selected_organ]
    selected_protein = st.sidebar.selectbox('Select Protein', proteins)
    selected_model = st.sidebar.selectbox('Select Model', models)
    smiles_string = st.sidebar.text_input('Enter SMILES String')

    # Main panel - Display Model Flowchart
    st.write("## Model Architecture")
    display_mermaid_flowchart(selected_model)

    # Predict Button
    if st.button('Predict'):
        try:
            model = load_model(selected_model, selected_protein)
            graph = smiles_to_graph(smiles_string)

            if graph is not None:
                model.eval()
                with torch.no_grad():
                    prediction = model(graph)
                    predicted_score = prediction.item()
                st.success(f'Predicted Docking Score: {predicted_score}')
            else:
                st.error('Invalid SMILES string')
        except Exception as e:
            st.error(f'An error occurred: {e}')

    # Additional Information Section
    st.write("## Additional Information")
    st.markdown("Here you can add more information about the project, model, or any other relevant details.")

if __name__ == '__main__':
    main()

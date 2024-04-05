import streamlit as st

# Preset data for dropdown menus
organs = {
    'Brain': ['O14672', 'P07900', 'P35869', 'P40763', 'P49841', 'Q9UBS5l', 'Q00535', 'Q11130', 'Q16539', 'P05129'], 
    'Liver': ['P04150', 'P14555', 'P19793', 'P07900_Liver', 'P22845', 'P42574', 'P55210', 'Q15465', 'P35869_Liver', 'Q96RI1'],
    'Kidney': ['O14920', 'P12821', 'P35869_Kidney', 'P42574_Kidney', 'P55210_Kidney', 'Q15303', 'Q16236', 'Q16665', 'P41595','P80365']
}

def main():
    st.title('ProteoDockNet: A Graph Neural Network Based Platform for Docking Score Prediction')

    # Load the name mapping dictionary from the CSV file
    name_mapping = {'P14555': 'Phospholipase A2, membrane associated', 'Q16539': 'Mitogen-activated protein kinase 14', 'P35869': 'Aryl hydrocarbon receptor', 'Q96RI1': 'Bile acid receptor', 'P41595': '5-hydroxytryptamine receptor 2B', 'P40763': 'Signal transducer and activator of transcription 3', 'Q16665': 'Hypoxia-inducible factor 1-alpha', 'Q16236': 'Nuclear factor erythroid 2-related factor 2', 'Q00535': 'Cyclin-dependent kinase 5', 'O14672': 'Disintegrin and metalloproteinase domain-containing protein 10', 'P07900': 'Heat shock protein HSP 90-alpha', 'P49841': 'Glycogen synthase kinase-3 beta', 'Q15465': 'Sonic hedgehog protein', 'P05129': 'Protein kinase C gamma type', 'P04150': 'Glucocorticoid receptor', 'Q11130': 'Alpha-(1,3)-fucosyltransferase 7', 'O14920': 'Inhibitor of nuclear factor kappa-B kinase subunit beta', 'P80365': '11-beta-hydroxysteroid dehydrogenase type 2'}


    # Sidebar for user input
    st.sidebar.header("User Input Features")
    selected_organ = st.sidebar.selectbox('Select Organ', list(organs.keys()))
    proteins = organs[selected_organ]
    selected_protein = st.sidebar.selectbox('Select Protein', proteins)
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
            selected_protein_name = name_mapping.get(selected_protein, selected_protein)  # Use the mapped name if available, else use original UniProt ID
            if selected_model == 'GCN':
                model = load_gcn_model(selected_protein_name)  # Using the mapped name
            elif selected_model == 'GCN+GAT':
                model = load_gcn_gat_model(selected_protein_name)  # Using the mapped name
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

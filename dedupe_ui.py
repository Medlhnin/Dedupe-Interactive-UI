import streamlit as st
import pandas as pd
import dedupe
from io import StringIO
import random

# Add encoding options (common encodings)
ENCODINGS = [
    "utf-8", 
    "latin-1", 
    "iso-8859-1", 
    "utf-16",
    "windows-1252",
    "ascii"
]

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'deduper' not in st.session_state:
    st.session_state.deduper = None
if 'labeled_pairs' not in st.session_state:
    st.session_state.labeled_pairs = {"match": [], "distinct": []}
if 'training_pairs' not in st.session_state:
    st.session_state.training_pairs = []

# Streamlit app
st.title("Dedupe Interactive UI")

# Step 1: Upload CSV File with Encoding Selection
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    # Let user select encoding
    selected_encoding = st.selectbox(
        "Select file encoding (if unsure, try 'latin-1' or 'utf-8'):",
        ENCODINGS,
        index=0  # Default to utf-8
    )

    try:
        # Read CSV with selected encoding
        df = pd.read_csv(uploaded_file, encoding=selected_encoding)
        
        # Convert data to dictionary of dictionaries
        data_dict = {idx: record for idx, record in enumerate(df.to_dict('records'))}
        st.session_state.data = data_dict
        st.success("File loaded successfully!")
        
        # Show preview
        st.subheader("Data Preview")
        st.write(df.head())

        # Step 2: Automatically Select All Fields
        selected_fields = df.columns.tolist()  # Use all columns
        st.write(f"Selected fields for deduplication: {selected_fields}")

        # Initialize Dedupe with variable objects
        variables = [dedupe.variables.String(field) for field in selected_fields]
        deduper = dedupe.Dedupe(variables)
        st.session_state.deduper = deduper
        
        # Step 3: Prepare Training Pairs
        if not st.session_state.training_pairs:
            deduper.prepare_training(st.session_state.data)
            st.session_state.training_pairs = deduper.uncertain_pairs()
            random.shuffle(st.session_state.training_pairs)
        
        # Step 4: Interactive Labeling
        st.subheader("Label Record Pairs")
        if st.session_state.training_pairs:
            pair = st.session_state.training_pairs.pop()
            record_1, record_2 = pair
            
            # Display pair for labeling
            col1, col2 = st.columns(2)
            with col1:
                st.write("Record 1:")
                st.json(record_1)
            with col2:
                st.write("Record 2:")
                st.json(record_2)
            
            # Buttons for labeling
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Match"):
                    st.session_state.labeled_pairs["match"].append((record_1, record_2))
                    st.success("Labeled as Match!")
            with col2:
                if st.button("❌ Not a Match"):
                    st.session_state.labeled_pairs["distinct"].append((record_1, record_2))
                    st.success("Labeled as Not a Match!")
            with col3:
                if st.button("⏭ Skip"):
                    st.session_state.training_pairs.append(pair)
                    st.info("Pair skipped.")
            
            st.write(f"Pairs remaining: {len(st.session_state.training_pairs)}")
            st.write(f"Pairs labeled (Match): {len(st.session_state.labeled_pairs['match'])}")
            st.write(f"Pairs labeled (Distinct): {len(st.session_state.labeled_pairs['distinct'])}")
        else:
            st.warning("No more pairs to label. You can proceed to training.")

        # Step 5: Train Model (User-Triggered)
        if st.button("Proceed to Training"):
            if len(st.session_state.labeled_pairs["match"]) + len(st.session_state.labeled_pairs["distinct"]) > 0:
                # Format labeled pairs for Dedupe
                labeled_pairs_formatted = {
                    "match": st.session_state.labeled_pairs["match"],
                    "distinct": st.session_state.labeled_pairs["distinct"]
                }
                
                # Add labeled data to Dedupe
                st.session_state.deduper.mark_pairs(labeled_pairs_formatted)
                
                # Train the model
                st.session_state.deduper.train()
                
                # Step 6: Clustering and Results
                clustered_records = st.session_state.deduper.partition(st.session_state.data)
                
                # Display Results
                st.subheader("Duplicate Clusters")
                for cluster_id, cluster in enumerate(clustered_records):
                    st.write(f"Cluster {cluster_id + 1}:")
                    for record in cluster:
                        st.json(record)
            else:
                st.error("No labeled pairs found. Label at least 10 pairs first.")

    except UnicodeDecodeError:
        st.error(f"Failed to decode with '{selected_encoding}' encoding. Try another encoding.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
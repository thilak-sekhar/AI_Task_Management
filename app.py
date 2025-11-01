import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Task Priority & Assignment System", page_icon="🤖", layout="centered")

st.title("🧠 AI Task Priority & Assignment System")

# --- File existence check ---
base_path = "C://Users//thila//Downloads//AI_TASK"

required_files = [
    "priority_prediction.pkl",
    "label_classification.pkl",
    "Synthetic Data.csv",
    "title_embeddings.csv",
    "description_embeddings.csv"
]

missing = [f for f in required_files if not os.path.exists(os.path.join(base_path, f))]
if missing:
    st.error(f"❌ Missing files: {', '.join(missing)}. Please place them in {base_path}")
    st.stop()

# --- Load models and data safely ---
@st.cache_resource
def load_all():
    try:
        with open(os.path.join(base_path, 'priority_prediction.pkl'), 'rb') as f:
            priority_model = pickle.load(f)
        with open(os.path.join(base_path, 'label_classification.pkl'), 'rb') as f:
            label_model = pickle.load(f)
        df = pd.read_csv(os.path.join(base_path, "Synthetic Data.csv"))
        title_emb_df = pd.read_csv(os.path.join(base_path, "title_embeddings.csv"))
        desc_emb_df = pd.read_csv(os.path.join(base_path, "description_embeddings.csv"))
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return priority_model, label_model, df, title_emb_df, desc_emb_df, model
    except Exception as e:
        st.error(f"⚠️ Error loading models/data: {e}")
        st.stop()

priority_model, label_model, df, title_emb_df, desc_emb_df, embedder = load_all()
st.success("✅ Models and data loaded successfully!")

# --- User Input ---
st.subheader("📝 Enter Task Details")
title_text = st.text_input("Task Title", "fix error in the login page")
desc_text = st.text_area("Task Description", "errors in the login page are to be fixed")

if st.button("🔍 Predict Task Details"):
    try:
        with st.spinner("Analyzing task... Please wait..."):
            combined_text = title_text + " " + desc_text

            # Embeddings
            title_emb = embedder.encode([title_text])
            desc_emb = embedder.encode([desc_text])
            combined_emb = np.concatenate((title_emb, desc_emb), axis=1)

            # Label prediction
            label_encoder = LabelEncoder()
            label_encoder.fit(df["Label"].astype(str))

            label_pred = label_model.predict(combined_emb)
            predicted_label = label_encoder.inverse_transform(label_pred.astype(int))[0]

            # Priority prediction
            predicted_priority = priority_model.predict([combined_text])[0]

            # --- Display Results ---
            st.success("✅ Prediction Results")
            st.write(f"**Predicted Label:** {predicted_label}")
            st.write(f"**Predicted Priority:** {predicted_priority}")

            # --- Task Assignment ---
            data = df.copy()
            data['DueDate'] = pd.to_datetime(data['DueDate'], errors='coerce')
            data['CreatedAt'] = pd.to_datetime(data['CreatedAt'], errors='coerce')
            data = data.dropna(subset=['DueDate', 'Workload'])

            sorted_df = data.sort_values(by=['Workload', 'DueDate'], ascending=[True, False])
            best_assignee = sorted_df.iloc[0]['Assignee']
            best_workload = sorted_df.iloc[0]['Workload']
            best_due = sorted_df.iloc[0]['DueDate']

            st.subheader("👤 Recommended Assignee")
            st.write(f"**Assign To:** {best_assignee}")
            st.write(f"**Current Workload:** {best_workload}")
            st.write(f"**Next Due Date:** {best_due.date() if pd.notna(best_due) else 'N/A'}")

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

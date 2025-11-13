import os
import streamlit as st
from neo4j import GraphDatabase
from dotenv import load_dotenv
from utility.common import duck_db_parquet_analysis, gen_live_data , neo4j_analysis, llm_func
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import joblib
import pandas as pd
import shap
import faiss
import pickle


st.set_page_config(layout="wide", page_title="Fin-Fraud Detection", page_icon=":anchor:")
st.markdown(f"""
    <style>
 
    .stApp {{
        font-family: 'Segoe UI', sans-serif;
        padding: 1rem 2rem;
    }}

    /* Rounded widgets */
    .stButton>button {{
        border-radius: 10px;
        background-color: #b5fc03;
        color: black;
    }}

    .stTextInput>div>div>input {{
        border-radius: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

st.title("📊 Financial Fraud Detection")

load_dotenv()  # load environment variables
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Live Data creation. To check with ML alogithms.
with st.sidebar:
    st.title("Live Data")
    number_of_records = st.sidebar.number_input("Enter number of records (<5):", min_value=1, max_value=5)
    if st.sidebar.button("Submit"):
        data_df = gen_live_data(number_of_records)
        data = data_df.to_dict(orient="records")
        for i, tx in enumerate(data):
            st.write(f"**Transaction {i+1}**")
            st.write(f"- amount: {tx['amount']}")
            st.write(f"- oldbalanceOrg: {tx['oldbalanceOrg']}")
            st.write(f"- newbalanceOrig: {tx['newbalanceOrig']}")
            st.write(f"- oldbalanceDest: {tx['oldbalanceDest']}")
            st.write(f"- newbalanceDest: {tx['newbalanceDest']}")
            st.markdown("---")
        # Add export as CSV option
        csv = data_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export as CSV",
            data=csv,
            file_name="live_data.csv",
            mime="text/csv"
        )

#Create intuitive tab layout
#Add pages as needed
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🦆 Duck DB Analysis", "🧠 Neo4j Data Analysis", "֎ Knowledge Graph", "🧮 Classical ML Model" , "🤖 LLM Query", "🔎 RAG Search"])

# Page 1
with tab1:
    duck_db_parquet_analysis()

with tab2:
    neo4j_analysis(driver)

with tab3:
    st.header("🧠 Knowledge Graph Visualization")
    # Query Neo4j for accounts and transactions
    def run_query(tx, query):
        return [dict(record) for record in tx.run(query)]

    query = '''
        MATCH (a1:Account)-[t:TRANSACTION]->(a2:Account)
        RETURN a1.id AS Sender, a2.id AS Receiver, round(t.amount, 2) as Amount, t.type as Type, t.isFraud as isFraud
        LIMIT 100
    '''
    with driver.session() as session:
        results = session.execute_read(run_query, query)

    # Build the knowledge graph
    G = nx.DiGraph()
    for row in results:
        sender = row['Sender']
        receiver = row['Receiver']
        amount = row['Amount']
        tx_type = row['Type']
        is_fraud = row['isFraud']
        G.add_node(sender, label="Account")
        G.add_node(receiver, label="Account")
        edge_label = f"{tx_type} | ${amount}"
        color = "red" if is_fraud else "#03fc7f"
        G.add_edge(sender, receiver, label=edge_label, color=color)

    # Visualize with pyvis
    net = Network(height="400px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    net.from_nx(G)
    net.repulsion(node_distance=200, central_gravity=0.33, spring_length=100, spring_strength=0.10, damping=0.95)
    net.save_graph("knowledge_graph.html")
    with open("knowledge_graph.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=450, scrolling=True)

    st.subheader("🔎 Ask a question about the graph (e.g., 'Show me all accounts with suspicious transactions')")
    user_query = st.text_input("Enter your question:", "Show me all accounts with suspicious transactions")
    submit_query = st.button("Run Query")

    if submit_query and user_query:
        cypher_query =  f'''
                MATCH (a:Account)-[t:TRANSACTION]->(b:Account)
                WHERE t.isFraud = true
                WITH a, count(t) AS suspicious_count
                WHERE suspicious_count >= 1
                RETURN a.id AS Account, suspicious_count AS FraudulentTransactions
                ORDER BY FraudulentTransactions DESC
                LIMIT 50
            '''
        if cypher_query:
            def run_query(tx, query):
                return [dict(record) for record in tx.run(query)]
            with driver.session() as session:
                results = session.execute_read(run_query, cypher_query)
            if results:
                st.success(f"Found {len(results)} accounts.")
                st.dataframe(results)
            # Visualize accounts in a simple graph (create graph ONCE, add all nodes, render ONCE)
                G = nx.Graph()
                for row in results:
                    acc_id = row['Account']
                    fraud_count = row.get('FraudulentTransactions', row.get('suspicious_count', ''))
                    label = f"{acc_id}"
                    title = f"Account: {acc_id}<br>Suspicious Txns: {fraud_count}"
                    G.add_node(acc_id, label=label, title=title, value=fraud_count)
                # Only create and render the pyvis Network ONCE
                net = Network(height="350px", width="100%", bgcolor="#222222", font_color="white")
                net.from_nx(G)
                net.save_graph("llm_query_graph.html")
                with open("llm_query_graph.html", "r", encoding="utf-8") as f:
                    html_content = f.read()
                components.html(html_content, height=370, scrolling=True)
            else:
                st.warning("No accounts found matching the query.")
        else:
            st.warning("Sorry, I can only answer queries about accounts with more than X suspicious transactions right now.")

with tab4:
    st.header("🧮 Classical ML Model Fraud Prediction")
    model_path = os.path.join("ML", "fraud_model.joblib")
    if not os.path.exists(model_path):
        st.error("Classical ML model not found at ML/fraud_model.joblib.")
    else:
        loaded = joblib.load(model_path)
        # Handle (model, scaler) tuple or just model
        if isinstance(loaded, tuple) and len(loaded) == 2:
            model, scaler = loaded
        else:
            model = loaded
            scaler = None
        st.write("Enter transaction details below or upload a CSV file for batch prediction.")
        input_method = st.radio("Choose input method:", ("Manual Input", "Upload CSV"))
        feature_names = [
            "step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "type_CASH_OUT", "type_TRANSFER"
        ]
        # SHAP explainer setup (TreeExplainer for tree models, KernelExplainer fallback)
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.Explainer(model)
        if input_method == "Manual Input":
            input_data = {}
            for feature in feature_names:
                if feature.startswith("type_"):
                    input_data[feature] = st.selectbox(f"{feature}", [0, 1], format_func=lambda x: "Yes" if x else "No")
                else:
                    default_val = 1.0 if feature == "step" else 0.0
                    input_data[feature] = st.number_input(f"{feature}", value=default_val, min_value=0.0)
            if st.button("Predict Fraud (Manual)"):
                X = pd.DataFrame([input_data])
                if scaler is not None:
                    # Only scale the columns the scaler was fit on
                    if hasattr(scaler, 'feature_names_in_'):
                        scale_cols = list(scaler.feature_names_in_)
                    else:
                        # Fallback: assume first n_features_in_ columns are numeric
                        scale_cols = [col for col in X.columns if col not in ["type_CASH_OUT", "type_TRANSFER"]][:scaler.n_features_in_]
                    X_to_scale = X[scale_cols]
                    X_scaled = scaler.transform(X_to_scale)
                    X_scaled_df = pd.DataFrame(X_scaled, columns=scale_cols, index=X.index)
                    # Combine scaled and unscaled columns
                    X_rest = X.drop(columns=scale_cols)
                    X_model = pd.concat([X_scaled_df, X_rest], axis=1)[feature_names]
                else:
                    X_model = X
                pred = model.predict(X_model)[0]
                st.success(f"Prediction: {'FRAUD' if pred == 1 else 'NOT FRAUD'}")
                # SHAP explanation
                shap_values = explainer.shap_values(X_model)
                st.subheader("Feature impact for this prediction:")
                shap.initjs()
                st.pyplot(shap.plots._waterfall.waterfall_legacy(
                    explainer.expected_value[1] if isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value,
                    shap_values[1][0] if isinstance(shap_values, list) else shap_values[0],
                    X.iloc[0], show=False))
        else:
            uploaded_file = st.file_uploader("Upload CSV file with columns: " + ", ".join(feature_names), type=["csv"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                if all(f in df.columns for f in feature_names):
                    if scaler is not None:
                        if hasattr(scaler, 'feature_names_in_'):
                            scale_cols = list(scaler.feature_names_in_)
                        else:
                            scale_cols = [col for col in df[feature_names].columns if col not in ["type_CASH_OUT", "type_TRANSFER"]][:scaler.n_features_in_]
                        X_to_scale = df[feature_names][scale_cols]
                        X_scaled = scaler.transform(X_to_scale)
                        X_scaled_df = pd.DataFrame(X_scaled, columns=scale_cols, index=df.index)
                        X_rest = df[feature_names].drop(columns=scale_cols)
                        X_model = pd.concat([X_scaled_df, X_rest], axis=1)[feature_names]
                    else:
                        X_model = df[feature_names]
                    preds = model.predict(X_model)
                    df['Prediction'] = ["FRAUD" if p == 1 else "NOT FRAUD" for p in preds]
                    st.dataframe(df)
                    st.markdown("---")
                    row_idx = st.number_input("Select row index to explain (0-based):", min_value=0, max_value=len(df)-1, value=0)
                    X_row = df[feature_names].iloc[[row_idx]]
                    if scaler is not None:
                        if hasattr(scaler, 'feature_names_in_'):
                            scale_cols = list(scaler.feature_names_in_)
                        else:
                            scale_cols = [col for col in X_row.columns if col not in ["type_CASH_OUT", "type_TRANSFER"]][:scaler.n_features_in_]
                        X_to_scale = X_row[scale_cols]
                        X_scaled = scaler.transform(X_to_scale)
                        X_scaled_df = pd.DataFrame(X_scaled, columns=scale_cols, index=X_row.index)
                        X_rest = X_row.drop(columns=scale_cols)
                        X_row_model = pd.concat([X_scaled_df, X_rest], axis=1)[feature_names]
                    else:
                        X_row_model = X_row
                    shap_values = explainer.shap_values(X_row_model)
                    st.subheader(f"Feature impact for row {row_idx}:")
                    shap.initjs()
                    st.pyplot(shap.plots._waterfall.waterfall_legacy(
                        explainer.expected_value[1] if isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value,
                        shap_values[1][0] if isinstance(shap_values, list) else shap_values[0],
                        X_row.iloc[0], show=False))
                else:
                    st.error("CSV missing required columns.")

with tab5:
    # Keep existing LLM features
    llm_func(driver)

with tab6:
    st.header("🔎 RAG Search (FAISS)")
    # Load FAISS index and docstore
    @st.cache_resource
    def load_faiss_and_docstore():
        index = faiss.read_index("faiss.index")
        with open("docstore.pkl", "rb") as f:
            docstore = pickle.load(f)
        return index, docstore
    index, docstore = load_faiss_and_docstore()

    # Dummy embed function (replace with your real embedding model)
    def embed_query(text):
        # Example: use a simple hash for demo; replace with real embedding
        import numpy as np
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.rand(index.d)

    user_query = st.text_input("Enter your RAG query:", value="Which account had the highest amount transaction and is marked as fraud?")
    top_k = st.number_input("Top K results", min_value=1, max_value=10, value=3)
    if st.button("Search RAG") and user_query:
        query_vec = embed_query(user_query).reshape(1, -1)
        D, I = index.search(query_vec, top_k)
        st.subheader("Top Retrieved Results:")
        for rank, idx in enumerate(I[0]):
            doc = docstore.get(idx, "[Not found]")
            st.markdown(f"**Rank {rank+1}:** {doc}")

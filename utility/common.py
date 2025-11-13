import base64
import glob
import json
import os
import streamlit as st
import altair as alt
import pandas as pd
import duckdb
from openai import OpenAI
from sdv.single_table import GaussianCopulaSynthesizer , CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import Metadata

def get_base64_image(img):
    with open(img, "rb") as imgFile:
        return base64.b64encode(imgFile.read()).decode()

#---------------------------------------------------------------------------------------------------------------

def gen_live_data(number_of_records):
    data = pd.read_csv(r"C:\Users\Dinesh Narayana\Downloads\Fraud Detection Project\archive\sample_csv.txt")
    metadata = Metadata.detect_from_dataframe(data)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    return synthesizer.sample(num_rows=number_of_records)

#---------------------------------------------------------------------------------------------------------------

#Streamlit table or DF writes the data not aligned. Using HTML styled table in these cases.
def generate_styled_html_table(data):

    if not data:
        return "<p style='color:gray;'>No data to display.</p>"

    headers = data[0].keys()
    html = """
    <style>
        .styled-table {
            width: 100%;
            border-collapse: collapse;
            text-align: center;
        }
        .styled-table th, .styled-table td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        .styled-table thead th {
            position: sticky;
            top: 0;
            background-color: #f2f2f2;
            z-index: 1;
        }
        .highlight {
            background-color: #ffe6e6;
            font-weight: bold;
            color: red;
        }
    </style>
    <table class="styled-table">
        <thead>
            <tr>
    """
    # Header row
    html += "".join([f"<th>{h}</th>" for h in headers])
    html += "</tr></thead><tbody>"

    # Data rows with conditional styling
    for row in data:
        amount = row.get("Amount", 0)
        highlight_class = "highlight" if amount > 200000 else ""
        html += "<tr>"
        for h in headers:
            cell_value = row[h]
            html += f"<td class='{highlight_class}'>{cell_value}</td>"
        html += "</tr>"

    html += "</tbody></table>"
    return html

#-----------------------------------------------------------------------------------------------------

def duck_db_parquet_analysis():
    # Parquet folder path
    DATA_PATH = r"C:\Users\Dinesh Narayana\Downloads\Fraud Detection Project\output_data"
    TABLE_NAME = "transactions"

    con = duckdb.connect(database=':memory:')
    parquet_files = glob.glob(os.path.join(DATA_PATH, "*.parquet"))

    if not parquet_files:
        st.warning("No Parquet files found in the data directory.")
    else:
        con.execute(f"""
        CREATE VIEW {TABLE_NAME} AS 
        SELECT * FROM read_parquet('{DATA_PATH}/*.parquet')
        """)

    st.subheader("🚨 Data highlights")
    account_count_orig = con.execute("SELECT COUNT(distinct nameOrig) FROM transactions").fetchone()[0]
    account_count_dest = con.execute("SELECT COUNT(distinct nameDest) FROM transactions").fetchone()[0]
    transactional_cnt = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    fraud_count = con.execute("SELECT COUNT(*) FROM transactions WHERE isFraud = 1").fetchone()[0]
    # st.metric("Total Frauds", fraud_count)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Accounts", f"{(account_count_orig + account_count_dest):,}")
    col2.metric("Number of Transactions", f"{transactional_cnt:,}")
    col3.metric("Total number Fradulent transactions", f"{fraud_count:,}")
    st.divider()

    st.subheader("💰 Total Transaction Amount per Type")
    amount_by_type = con.execute("""
        SELECT type, SUM(amount) AS total_amount
        FROM transactions
        GROUP BY type
        ORDER BY total_amount DESC
        """).df()

    chart = alt.Chart(amount_by_type.reset_index()).mark_bar().encode(
        x='type',
        y='total_amount',
        color='type'
    ).properties(width=600, height=400)

    st.altair_chart(chart, use_container_width=True)
    st.divider()

    # Avg amount by type
    st.subheader("📊 Average Transaction Amount by Type")
    avg_amount = con.execute("""
        SELECT type as Type, ROUND(AVG(amount), 2) AS 'Average Amount'
        FROM transactions
        GROUP BY type
        """).df()
    table_html = generate_styled_html_table(avg_amount.to_dict(orient="records"))
    st.markdown(table_html, unsafe_allow_html=True)
    st.divider()

    # Fraud types
    st.subheader("📌 Fraud Types Count")
    fraud_types = con.execute("""
        SELECT type, COUNT(*) AS count
        FROM transactions
        WHERE isFraud = 1
        GROUP BY type
        """).df()
    chart = alt.Chart(fraud_types).mark_bar().encode(
        x='type',
        y='count',
        color='type',
        tooltip=['count', 'type']
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)
    st.divider()

    # Top fraud customers
    st.subheader("🧑‍💻 Top Fraudulent Customers")
    top_customers = con.execute("""
        SELECT nameOrig, COUNT(*) as frauds
        FROM transactions
        WHERE isFraud = 1
        GROUP BY nameOrig
        ORDER BY frauds DESC
        LIMIT 5
        """).df()
    table_html = generate_styled_html_table(top_customers.to_dict(orient="records"))
    st.markdown(table_html, unsafe_allow_html=True)
    st.divider()

#---------------------------------------------------------------------------------------------------------------

def neo4j_analysis(driver):
    # query executor
    def run_query(tx, query):
        return [dict(record) for record in tx.run(query)]

    g_query = {
        "Sample Transactions (Limit 100)": """
        MATCH (a1:Account)-[t:TRANSACTION]->(a2:Account)
        RETURN a1.id AS Sender, a2.id AS Receiver,round(t.amount, 2) as Amount, t.type as Type, t.isFraud as isFraud
        LIMIT 100
    """,
        "Fraudulent Transactions": """
        MATCH (a1:Account)-[t:TRANSACTION {isFraud: true}]->(a2:Account)
        RETURN a1.id AS Fraudster, a2.id AS Victim, round(t.amount, 2) as Amount, t.type as Type
    """,
        "Transaction Type Fraud Count": """
        MATCH (:Account)-[t:TRANSACTION]->(:Account)
        WHERE t.isFraud = true
        RETURN t.type AS TransactionType, COUNT(*) AS FraudCount;
    """,
        "Accounts That Sent Fraudulent Transfers": """
        MATCH (a:Account)-[t:TRANSACTION {isFraud: true}]->()
        RETURN DISTINCT a.id AS Fraudster, count(t) AS fraud_count
        ORDER BY fraud_count DESC;
    """,
        "Top Transaction Amounts": """
        MATCH (a1:Account)-[t:TRANSACTION]->(a2:Account)
        RETURN a1.id AS Sender, a2.id AS Receiver, round(t.amount, 2) as Amount, t.isFraud as isFraud
        ORDER BY t.amount DESC
        LIMIT 10;
    """,
        "Top 5 Accounts with high number of transactions": """
        MATCH (a:Account)-[t:TRANSACTION]->()
        RETURN a.id AS Account, COUNT(*) AS tx_count
        ORDER BY tx_count DESC
        LIMIT 5
    """,
        "Account Pairs with Multiple Transactions Between Them": """
        MATCH (a1:Account)-[t:TRANSACTION]->(a2:Account)
        WITH a1, a2, count(*) AS tx_count
        WHERE tx_count >= 1
        RETURN a1.id AS Sender, a2.id AS Receiver, tx_count
        ORDER BY tx_count DESC;
    """,
        "Count of Fraud by Step": """
        MATCH ()-[t:TRANSACTION {isFraud: true}]->()
        RETURN t.step AS TimeStep, COUNT(*) AS FraudCount
        ORDER BY TimeStep
    """,
    }

    selection = st.selectbox("Choose a query", list(g_query.keys()))

    with driver.session() as session:
        with st.spinner("Querying DB...", show_time=True):
            results = session.execute_read(run_query, g_query[selection])
            st.subheader("🔎 Query Results")
            table_html = generate_styled_html_table(results)
            st.markdown(table_html, unsafe_allow_html=True)


    # if selection == "Transaction Type Fraud Count":
    #             if df.empty:
    #                 print("Nothing to show")
    #             else:
    #                 st.bar_chart(df.set_index("type"))
#---------------------------------------------------------------------------------------------------------
def llm_func(driver):
    st.title("Ask LLM")

    def llm_cypher_query(account_id: str):
        query = f"""
                MATCH (a:Account)-[t:TRANSACTION]->(b:Account)
                WHERE a.id = '{account_id}' OR b.id = '{account_id}'
                RETURN a.id AS Sender, b.id AS Receiver, round(t.amount, 2) AS Amount, t.type AS Type, 
                t.isFraud AS IsFraud, t.isFlaggedFraud AS IsFlaggedFraud
                LIMIT 50
                """
        with driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    st.caption("Note: Both Sender/Receiver transactions will be fetched for the given account")
    user_input = st.text_input("Enter an Account ID", placeholder="C1305486145")

    if st.button("Submit") and user_input:
        with st.spinner("Fetching data from Neo4j DB...", show_time=True):
            cypher_op = llm_cypher_query(user_input.strip())
            df_llm = pd.DataFrame(cypher_op)
            st.subheader(f"🔎 Rows Returned for Account ID : {user_input}")
            table_html = generate_styled_html_table(cypher_op)
            st.markdown(table_html, unsafe_allow_html=True)
            # st.table(df_llm)
            st.divider()

        client = OpenAI(
            base_url="http://localhost:12434/engines/llama.cpp/v1/",  # llama.cpp server
            api_key="llama"
        )

        try:
            if cypher_op:
                context_text = json.dumps(cypher_op, indent=2)
                with st.spinner("Waiting for LLM response...", show_time=True):
                    response = client.chat.completions.create(
                        model="ai/mistral",  # model name from docker
                        messages=[{"role": "user", "content": f"Based on the following transactions, explain any signs of fraud:\n{context_text}"}],
                        temperature=0.7,
                        max_tokens=256
                    )

                    st.markdown("### Response:")
                    st.write(response.choices[0].message.content)

                    rating_prompt = f"""
                            You previously gave this explanation for the transaction context below.
                            ### Your Response:
                            {response.choices[0].message.content}
                            Now, rate the quality and clarity of your explanation **on a scale from 0.0 to 1.0**, where:
                            - 1.0 = Excellent explanation, clearly highlights fraud indicators.
                            - 0.0 = Useless or totally irrelevant.
                            - Respond only with a single number like: `0.85`
                            """

                    rating_response = client.chat.completions.create(
                        model="ai/mistral",
                        messages=[{"role": "user", "content": rating_prompt}],
                        temperature=0.0,
                        max_tokens=10  # keep it small since we expect only a score
                    )
                    st.markdown("### Rating Response:")
                    st.write(rating_response.choices[0].message.content)

            else:
                st.warning("No data found for that account.")
        except Exception as e:
            st.error(f"Error connecting to LLM: {e}")

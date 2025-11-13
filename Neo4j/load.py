import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Load .env variables
load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# node (account) and relationship (transaction between accounts)
#Use UNWIND for faster loading
#Create batches whilst loading
#Create indexes in Neo4j instance
def create_account_nodes(tx, account_ids):
    tx.run("""
        UNWIND $accounts AS account_id
        MERGE (:Account {id: account_id})
    """, accounts=account_ids)


def create_transaction_relationships(tx, transactions):
    tx.run("""
        UNWIND $txs AS row
        MATCH (a:Account {id: row.nameOrig})
        MATCH (b:Account {id: row.nameDest})
        MERGE (a)-[:TRANSACTION {
            step: row.step,
            type: row.type,
            amount: row.amount,
            oldbalanceOrg: row.oldbalanceOrg,
            newbalanceOrig: row.newbalanceOrig,
            oldbalanceDest: row.oldbalanceDest,
            newbalanceDest: row.newbalanceDest,
            isFraud: row.isFraud,
            isFlaggedFraud: row.isFlaggedFraud
        }]->(b)
    """, txs=transactions)


def main():
    # Load CSV
    df = pd.read_csv(r"C:\Users\Dinesh Narayana\Downloads\Fraud Detection Project\archive\Financial_datasets_log.csv")

    # Ensure proper types
    df["isFraud"] = df["isFraud"].astype(bool)
    df["isFlaggedFraud"] = df["isFlaggedFraud"].astype(bool)

    # Extract unique accounts
    accounts = list(set(df["nameOrig"]).union(set(df["nameDest"])))

    # Convert entire DataFrame to list of dicts
    transactions = df.to_dict(orient="records")

    # --- Batch Loading in 1000's ---
    with driver.session() as session:
        for i in range(0, len(accounts), 1000):
            session.execute_write(create_account_nodes, accounts[i:i+1000])

        for i in range(0, len(transactions), 500):
            session.execute_write(create_transaction_relationships, transactions[i:i+500])

    driver.close()
    print("Neo4j Data load completed successfully.")


if __name__ == "__main__":
    main()

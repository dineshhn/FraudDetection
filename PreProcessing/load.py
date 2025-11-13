import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def create_account_nodes(tx, account_id):
    tx.run("""
        MERGE (:Account {id: $account_id})
    """, account_id=account_id)


def create_transaction_relationship(tx, row):
    tx.run("""
        MATCH (a:Account {id: $nameOrig})
        MATCH (b:Account {id: $nameDest})
        MERGE (a)-[t:TRANSACTION {
            step: $step,
            type: $type,
            amount: $amount,
            oldbalanceOrg: $oldbalanceOrg,
            newbalanceOrig: $newbalanceOrig,
            oldbalanceDest: $oldbalanceDest,
            newbalanceDest: $newbalanceDest,
            isFraud: $isFraud,
            isFlaggedFraud: $isFlaggedFraud
        }]->(b)
    """, {
        "nameOrig": row["nameOrig"],
        "nameDest": row["nameDest"],
        "step": int(row["step"]),
        "type": row["type"],
        "amount": float(row["amount"]),
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),
        "isFraud": bool(row["isFraud"]),
        "isFlaggedFraud": bool(row["isFlaggedFraud"]),
    })


def main():
    df = pd.read_csv(r"C:\Users\Dinesh Narayana\Downloads\Fraud Detection Project\archive\sample_csv.txt")

    with driver.session() as session:
        # Load nodes
        accounts = set(df["nameOrig"]).union(set(df["nameDest"]))
        for acc_id in accounts:
            session.execute_write(create_account_nodes, acc_id)

        # Load relationships
        for _, row in df.iterrows():
            session.execute_write(create_transaction_relationship, row)

    driver.close()
    print("All nodes and relationship loaded successfully.")


if __name__ == "__main__":
    main()

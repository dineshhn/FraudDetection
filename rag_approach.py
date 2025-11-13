# standalone script. Run the script before executing streamlit job.

import numpy as np
import faiss
import pickle
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os
import argparse

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

def get_neo4j_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return driver

def get_account_ids(driver, limit=100):
    query = f"""
    MATCH (a:Account)
    RETURN DISTINCT a.id AS account_id
    LIMIT {limit}
    """
    with driver.session() as session:
        result = session.run(query)
        return [r['account_id'] for r in result]

def get_transactions(driver, account_id=None, limit=4000):
    if account_id and account_id != 'All':
        query = f"""
        MATCH (a:Account)-[t:TRANSACTION]->(b:Account)
        WHERE a.id = '{account_id}' OR b.id = '{account_id}'
        RETURN a.id AS nameOrig, b.id AS nameDest, t.amount AS amount, t.isFraud AS isFraud, id(t) AS tx_id
        LIMIT {limit}
        """
    else:
        query = f"""
        MATCH (a:Account)-[t:TRANSACTION]->(b:Account)
        RETURN a.id AS nameOrig, b.id AS nameDest, t.amount AS amount, t.isFraud AS isFraud, id(t) AS tx_id
        LIMIT {limit}
        """
    with driver.session() as session:
        result = session.run(query)
        records = result.data()
    return records

def prepare_documents(transactions):
    docs = []
    for tx in transactions:
        nameOrig = tx.get('nameOrig') or "UnknownOrig"
        nameDest = tx.get('nameDest') or "UnknownDest"
        amount = tx.get('amount') or 0.0
        isFraud = tx.get('isFraud') or False
        tx_id = tx.get('tx_id', None)
        text = f"Transaction from {nameOrig} to {nameDest} and amount={amount} and is fraud={isFraud}"
        meta = {"nameOrig": nameOrig, "nameDest": nameDest, "amount": amount, "isFraud": isFraud, "tx_id": tx_id}
        docs.append(Document(page_content=text, metadata=meta))
    return docs

def build_faiss_index_ivf(docs, nlist=100):
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = [doc.page_content for doc in docs]
    embedded_vectors = embeddings_model.embed_documents(texts)
    vector_matrix = np.array(embedded_vectors).astype('float32')
    d = vector_matrix.shape[1]
    num_vectors = vector_matrix.shape[0]
    index_to_docstore = {i: doc for i, doc in enumerate(docs)}
    if num_vectors < 2:
        index = faiss.IndexFlatL2(d)
        index.add(vector_matrix)
    else:
        effective_nlist = min(nlist, num_vectors)
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, effective_nlist, faiss.METRIC_L2)
        index.train(vector_matrix)
        index.add(vector_matrix)
    return index, index_to_docstore, embeddings_model

def save_index(index, path="faiss.index"):
    faiss.write_index(index, path)

def load_index(path="faiss.index"):
    return faiss.read_index(path)

def save_docstore(docstore, path="docstore.pkl"):
    with open(path, "wb") as f:
        pickle.dump(docstore, f)

def load_docstore(path="docstore.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index and docstore from Neo4j transactions.")
    parser.add_argument('--account', type=str, default='All', help='Account ID to index (or All)')
    parser.add_argument('--limit', type=int, default=4000, help='Max transactions to fetch')
    parser.add_argument('--nlist', type=int, default=100, help='FAISS nlist parameter')
    args = parser.parse_args()

    driver = get_neo4j_driver()
    print(f"Fetching transactions for account: {args.account}")
    transactions = get_transactions(driver, args.account, args.limit)
    print(f"Fetched {len(transactions)} transactions.")
    docs = prepare_documents(transactions)
    if not docs:
        print("No transactions found. Exiting.")
        return
    print("Building FAISS index...")
    index, docstore, _ = build_faiss_index_ivf(docs, nlist=args.nlist)
    save_index(index)
    save_docstore(docstore)
    print(f"Saved FAISS index and docstore for {len(docs)} documents.")

if __name__ == "__main__":
    main()

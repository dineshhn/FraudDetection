import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components

from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

@st.cache_data
def get_fraud_graph_data():
    with driver.session() as session:
        # Adjust Cypher to match your actual schema
        cypher = """
        MATCH (a:Account)-[r:TRANSACTION]->(b:Account)
        RETURN a, r, b
        LIMIT 100
        """
        result = session.run(cypher)

        nodes = {}
        edges = []

        for record in result:
            a = record["a"]
            b = record["b"]
            r = record["r"]

            for node in [a, b]:
                node_id = node.element_id
                if node_id not in nodes:
                    nodes[node_id] = {
                        "id": node_id,
                        "label": node["name"] if "name" in node else f"Account {node_id}",
                        "title": str(dict(node)),
                        "color": "#00cc88"  # default green for accounts
                    }

            # Color and label the edge
            is_fraud = r.get("isFraud", False)
            amount = r.get("amount", 0.0)

            edge_color = "#FF4136" if is_fraud else "#cccccc"
            edge_label = f"${amount:.2f}" + (" ⚠️ FRAUD" if is_fraud else "")

            edges.append({
                "source": a.element_id,
                "target": b.element_id,
                "label": edge_label,
                "color": edge_color
            })

        return list(nodes.values()), edges

# Fetch and build the graph
nodes, edges = get_fraud_graph_data()
net = Network(height="600px", width="100%", bgcolor="#111", font_color="white", directed=True)

# Add nodes
for node in nodes:
    net.add_node(
        node["id"],
        label=node["label"],
        title=node["title"],
        color=node["color"]
    )

# Add edges
for edge in edges:
    net.add_edge(
        edge["source"],
        edge["target"],
        label=edge["label"],
        color=edge["color"]
    )

# Display
net.repulsion(node_distance=180, central_gravity=0.2)
net.save_graph("fraud_graph.html")
components.html(open("fraud_graph.html", "r", encoding='utf-8').read(), height=650)

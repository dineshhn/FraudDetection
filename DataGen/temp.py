from langchain_neo4j import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

print(NEO4J_URI)
# Manually define schema string (adjust to match your actual Neo4j graph)
schema_str = """
Node types:
- Account(id: string)
- Transaction(id: string, amount: float, isFraud: boolean, step: int)

Relationships:
- (Account)-[:MADE]->(Transaction)
- (Transaction)-[:TO]->(Account)
"""

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

llm = ChatOpenAI(
    temperature=0,
    model="ai/mistral",  # match what your Llamafile container exposes
    base_url="http://localhost:12434/engines/llama.cpp/v1/",
    api_key="llama"  # llama.cpp ignores this, just needs to be non-empty
)

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt_template="""
You are an expert Cypher translator. Given the user question and schema, write an appropriate Cypher query.

Schema:
{schema}

Question:
{question}

Cypher Query:
""",
    schema=schema_str,  # Inject manual schema
    verbose=True
)

# Run it
question = "Show 5 recent suspicious transactions"
response = chain.run(question)
print(response)
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

graph_store = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    refresh_schema=True
)

llm = ChatOpenAI(
    temperature=0,
    model="ai/deepseek-r1-distill-llama",  # match what your Llamafile container exposes
    base_url="http://localhost:12434/engines/llama.cpp/v1/",
    api_key="llama"  # llama.cpp ignores this, just needs to be non-empty
)

schema_str = """
Node types:
- Account(id: string)
- Transaction(id: string, amount: float, isFraud: boolean, step: int)

Relationships:
- (Account)-[:MADE]->(Transaction)
- (Transaction)-[:TO]->(Account)
"""

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph_store,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt_template="""
You are an expert Cypher translator. Given the user question and schema, write an appropriate Cypher query.

Schema:
{schema}

Question:
{query}

Cypher Query:
""",
    schema=schema_str,
)

query = "Show 5 recent suspicious transactions"
response = chain.invoke({"query": "Show the top 10 largest fraudulent transactions"})
print(response)


#"Which users are connected to fraud?"
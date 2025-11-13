Step 1 : Get the Neo4j up and running with Docker
===============================================
This guide will help you set up a Neo4j database using Docker. Follow the steps below to get started.

Normal Setup:

docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -v neo4j_data:/data neo4j:latest

With plugins (APOC and GDS) enabled:

docker run -p 7474:7474 -p 7687:7687 --name neo4j-apoc -v neo4j_data:/data -e 'NEO4JLABS_PLUGINS=[\"apoc\"]' -e NEO4J_dbms_security_procedures_unrestricted=apoc.*,dbms.*,gds.* -e NEO4J_dbms_security_procedures_allowlist=apoc.*,dbms.*,gds.* neo4j:latest

Note : 
1. Make sure to replace 'neo4j:latest' with the specific version of Neo4j you want to use if needed. 
2. The volume 'neo4j_data' is used to persist the database data.

Accessing Neo4j:

url : http://localhost:7474/browser/  
user : neo4j  
password : neo4j

---------------------------------------
Step 2 : 
---------------------------------------

Get models ai/mistral and ai/deepseek-r1-distill-llama from docker container and keep it in local system. Make sure to get smaller models if you have less system memory.  

OpenAI Client Setup works in this case. Make sure to use correct model name when required.  

client = OpenAI(
base_url="http://localhost:12434/engines/llama.cpp/v1/",  # llama.cpp server
api_key="llama"
)

--------------------------------------- 

Step 3 : 
---------------------------------------
* Create a virtual environment and install required packages using requirements.txt
* Activate the virtual environment before running the scripts.
* Add all the environment variables in a .env file in the root directory.
---------------------------------------  
Step 4:
---------------------------------------
Run the following python scripts  

1. python .\rag_approach.py  : To create FAISS index
2. python .\convert2parquet.py : To convert CSV data to Parquet format
3. python .\load.py : To load data into Neo4j from csv files [creates nodes and relationships]
4. python .\ML\classicalML.py : Creates job.lib file for classical ML model [to be used in main script]
5. python .\st3.py : Main script to interact with Neo4j and get responses using RAG/LLM approach.



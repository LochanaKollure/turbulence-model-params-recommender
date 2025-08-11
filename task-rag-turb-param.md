# Retrieval Augmented Generation(RAG) system for parameter generator for a given CFD turbulence


## Context

It is needed to have a simple RAG based AI system that can
- retrieve relevant list of documents from a vector embedding database
- use LLM with a properly crafted system prompt for generating parameter values for a given turbulence model
- the LLM get the limited relevant document content retrieved from the vector embedding database
- where users can specify a turbulence model with additional text description about the task


The LLM with its system prompt should be specialized and expert in recommending values for model parameters of turbulence models.
The system needs to provide the ability for the user to select a turbulence model from a list of popular turbulence models. And then it should also include a list of model parameters for each model values for which the LLM needs to output in the exact expected structure.

The overall system will be built with Python with virtual env.

For the vector embedding database Pinecone needs to be used https://docs.pinecone.io/guides/get-started/quickstart.
PINECONE_API_KEY will be provided as an ENV variable.

Vector embedding and retrieval should be simple and lightweight.

To populate the vector embedding in the database so that the system can semantically query the relevant documents, there needs to be a script that populate/boostrap the database with a given list of articles URLs and PDF file URLs. Start with a couple of simple urls or pdf documents links. Proper documents list will be updated later.

Python streamlit will be used to build the UI needed for users to interact with the system.

OpenAI api with gpt-4o model will be used as the LLM, the OPENAI_API_KEY will always needs to be provided as an ENV variable.

Follow best practices with Python including having a git repository locally running setup.


## Task

Implement the RAG system needed to achieve turbulence model parameter recommendation for the turbulence model the user selects, considering the additional information the user provides and the relevant documents stored in vector embedding database. Use OpenAI gpt-4o LLM specialized in parameter value recommendation with provided information.

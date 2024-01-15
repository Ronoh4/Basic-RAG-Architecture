#This code demonstrates how to create a basic Retrieval-Augmented Generation (RAG) system using LlamaIndex and OpenAI's embedding model.

#Key Functionalities:
Loads text data from Wikipedia pages: Retrieves content from specified Wikipedia pages using the WikipediaReader loader.
Chunks text into nodes: Divides text into smaller segments for efficient indexing and retrieval.
Creates a vector store: Uses DeepLake to store text embeddings.
Generates text embeddings: Leverages OpenAI's embedding model to create vector representations of text.
Builds a searchable index: Indexes the text embeddings for fast retrieval.
Queries the index: Enables users to ask natural language questions and receive relevant responses based on the indexed content.

#Dependencies:
llama-index
dotenv
openai
activeloop-python-sdk (if using Activeloop for storage)

#Usage:
Install dependencies: pip install llama-index dotenv openai activeloop-python-sdk
Create a .env file: Store API keys and other sensitive information securely.
Run the code: Execute the Python script.
Query the index: Ask questions in natural language to retrieve relevant information.

#Additional Notes:
The code is intended for demonstration purposes and may require adjustments for specific use cases.
Refer to the LlamaIndex documentation for more advanced features and customization options.

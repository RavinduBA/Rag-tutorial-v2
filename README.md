# rag-tutorial-v2

As for the best practices, 

- We have to use the exact embedding function for two scenarios (Use the exact Vector embedding function for both ) for data stroing in vector database and for Querying the DB (Langchain has many different embedding functions , Refer Langchain documentation on embedding functions).

- There are various document loaders for different docuemnt types ( CSV, Markdown, Html, MS office, JSON ) in Langchain ,you can use an of those as document loader (Refer Langchian documentaion on document loaders).
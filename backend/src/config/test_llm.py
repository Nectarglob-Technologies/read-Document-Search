from config import Config

# Get LLM
llm = Config.get_llm()

# Call LLM
response = llm.invoke("Explain agentic RAG")

# Print result
print("\n=== RESPONSE ===\n")
print(response.content)
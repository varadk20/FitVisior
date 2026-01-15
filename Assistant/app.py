from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="mistral:7b-instruct",
    base_url="http://localhost:11434",
)

if __name__ == "__main__":
    print(llm.invoke("What is push pull leg?"))

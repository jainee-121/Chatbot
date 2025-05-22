from langchain_core.prompts import PromptTemplate

Template=PromptTemplate(
    template="""
    You are a helpful assistant. Use the provided context to answer the user's question.

Context:
{context}

Question: {question}
Answer:
""",
input_variables=['context','question'],
validate_template=True
)

Template.save('template.json')
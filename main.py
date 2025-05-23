from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_together import TogetherEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import load_prompt
from langchain_together import ChatTogether
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

loader=TextLoader("train.txt")
doc=loader.load()
text=doc[0].page_content

'''
The RecursiveCharacterTextSplitter tries to split your text at logical boundaries (like newlines, sentences, or paragraphs)
before falling back to character-based splitting.

If your text contains many newlines or is formatted with short lines (as in your train.txt),
the splitter will often break at those newlines, resulting in shorter chunks than your chunk_size (100 chars).

If you want logical, context-aware chunks, keep using RecursiveCharacterTextSplitter.
If you want fixed-size, overlapping chunks, switch to CharacterTextSplitter.
'''
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
splits = text_splitter.split_documents(doc)

embedding=TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
# docs=embedding.embed_documents(splits)#if you want to use splits here instead of passing it as doc pass as a string
# query=embedding.embed_query("what is the location of restaurant?")
# score=cosine_similarity([query],doc)[0]
# index, score = sorted(list(enumerate(score)),key=lambda x:x[1])[-1]
# print(splits[index],"\nsimilarity score:",score)

vectorstore=Chroma.from_documents(
    documents=splits,embedding=embedding)

template=load_prompt('template.json')
# prompt=template.invoke({
#     'context':'Grill Kitchen',
#     'question':'what is the menu of restaurant?'
# })
 
'''
The temperature parameter controls the randomness of the output.
if it is near 0 the answer will be determinstic.
if it is near 1 the answer will be more creative.
'''
llm= ChatTogether(model="meta-llama/Llama-3-70b-chat-hf",temperature=0.5)
chain= template | llm
answer=chain.invoke({
    'question':'what is the menu of restaurant?'
}) 

#simply qna messaging chatbot
# answer=llm.invoke(prompt)
# while True:
#     user_input=input("You :")
#     if user_input =="exit":
#         break
#     answer=llm.invoke(user_input)
#     print("AI :",answer.content)

#using history
# history=[]
# message=[
#     SystemMessage(content="you are a helpful ai assitant"),
#     HumanMessage(content="Hello")
# ]
# answer=llm.invoke(message)
# message.append(AIMessage(content=answer.content))
# print(message)

print(answer.content)

retriever = vectorstore.as_retriever()
# retrieved_documents = retriever.invoke("What is the name of the restaurant?")# gives semantic search w/o use of model 
# print(retrieved_documents[0].page_content) 
from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
llm=ChatTogether(model="meta-llama/Llama-3-70b-chat-hf")

template1=PromptTemplate(template='write a detailed report on {topic}',input_variables=['topic'])

template2=PromptTemplate(template='write a 5 line summary on following text. \n {text}',input_variables=['text'])

# w/o using chain
# prompt1= template1.invoke({'topic':"black hole"})
# result1=llm.invoke(prompt1)
# print(result1.content,"\n")

# prompt2=template2.invoke({'text':result1.content})
# result2=llm.invoke(prompt2)
# print(result2.content)

# using chain
parser=StrOutputParser()
chain=template1 | llm | parser | template2 | llm | parser
result=chain.invoke({'topic':"L J university"})
print(result)
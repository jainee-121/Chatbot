from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()
llm=ChatTogether(model="meta-llama/Llama-3-70b-chat-hf")

class Person(BaseModel):
    name: str=Field(description="name of the person")
    age : int=Field(description="age of the person")
    city: str=Field(description="name of the city the person belongs to")

parser=PydanticOutputParser(pydantic_object=Person)


template =PromptTemplate(template="generate the name age and city of a fictional {place} person 3 facts \n {format}",input_variables=['place'],partial_variables={'format':parser.get_format_instructions()})
# prompt=template.invoke({'place':"Indian"})
# result=llm.invoke(prompt)
# result=parser.parse(result.content)

# with chain
chain=template| llm | parser
result=chain.invoke({'place':'American'}) 
print(result,'\n',type(result))
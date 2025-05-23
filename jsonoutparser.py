from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
load_dotenv()
llm=ChatTogether(model="meta-llama/Llama-3-70b-chat-hf")

parser=JsonOutputParser()
template1=PromptTemplate(template='write a detailed report on {title} \n {topic}',input_variables=['title'],partial_variables={'topic':parser.get_format_instructions()})

# prompt=template1.format()

# result=llm.invoke(prompt)

# result=parser.parse(result.content)

# with chain
chain=template1 | llm | parser
result=chain.invoke({'title':"lion"})
print(type(result))
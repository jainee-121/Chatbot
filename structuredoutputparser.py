from dotenv import load_dotenv
from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
load_dotenv()
llm=ChatTogether(model="meta-llama/Llama-3-70b-chat-hf")

schema=[
    ResponseSchema(name='fact1',description='fact 1 about the topic'),
    ResponseSchema(name='fact2',description='fact 2 about the topic'),
    ResponseSchema(name='fact3',description='fact 3 about the topic')
]
parser=StructuredOutputParser.from_response_schemas(schema)
template =PromptTemplate(template="give 3 facts on {topic} \n {format}",input_variables=['topic'],partial_variables={'format':parser.get_format_instructions()})
# prompt=template.invoke({'topic':"lion"})
# result=llm.invoke(prompt)
# result=parser.parse(result.content)

# with chain
chain=template| llm | parser
result=chain.invoke({'topic':'lion'})
print(result,'\n',type(result))
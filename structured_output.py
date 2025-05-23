# for this code as together model doesnt provide supports structure output it cannot be used thats why the answer will be None.

from typing import TypedDict,Annotated,Literal,Optional
from dotenv import load_dotenv
from langchain_together import ChatTogether

load_dotenv()
llm=ChatTogether(model="mistralai/Mixtral-8x7B-Instruct-v0.1",temperature=0.5)

class Review(TypedDict):
    key_theme=Annotated[list[str],"Write down all the key themes discussed in the review in a list"]
    summary=Annotated[str,"A brief summary of the review"]
    sentiments=Annotated[Literal['neg','pos'],"Return sentiment of the review either negative, positive or neutral"]

# # pydantic is data validation and parsing lib in python.
# from pydantic import BaseModel,EmailStr,Field
# from typing import Optional, Literal
# class Review(BaseModel):
#     name: str ='nitish'
#     age: Optional[int] = None
#     email: EmailStr = Field(default=None , description="validate email")
#     rank: int = Field(ge=10)

# a={'name':'jainee','age':32,'email':'jainee121@gmail.com','rank':10}
# student=Review(**a)
# print(student.name)

structure=llm.with_structured_output(Review)
answer=structure.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
""")

print(answer)
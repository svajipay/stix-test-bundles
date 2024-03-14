
import os

from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

from bampy.extensions.langchain import LangChainInterface
from bampy.model import Credentials
from bampy.schemas import GenerateParams, ModelType

# make sure you have a .env file under bampy root with
# BAM_KEY=<your-bampy-key>
load_dotenv()
api_key = os.getenv("BAM_KEY", None)

params = GenerateParams(
    decoding_method="sample",
    max_new_tokens=100,
    min_new_tokens=1,
    stream=False,
    temperature=0.5,
    top_k=50,
    top_p=1,
).dict()  # Langchain uses dictionaries to pass kwargs

pt1 = PromptTemplate(input_variables=["topic"], template="Generate a random question about {topic}: Question: ")
pt2 = PromptTemplate(
    input_variables=["question"],
    template="Answer the following question: {question}",
)


creds = Credentials(api_key)
flan = LangChainInterface(model=ModelType.FLAN_UL2, credentials=creds, params=params)
bloomz = LangChainInterface(model=ModelType.BLOOMZ, credentials=creds)
prompt_to_flan = LLMChain(llm=flan, prompt=pt1)
flan_to_bloomz = LLMChain(llm=bloomz, prompt=pt2)
qa = SimpleSequentialChain(chains=[prompt_to_flan, flan_to_bloomz], verbose=True)
qa.run("life")

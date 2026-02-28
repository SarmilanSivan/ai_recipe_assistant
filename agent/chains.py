"""
AI configurations

"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from agent.prompts import extraction_prompt, sql_generation_prompt
from agent.schemas import ExtractionResult

from dotenv import load_dotenv
load_dotenv()

# Initialize the LLMs 
primary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
creative_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Bind structured outputs
structured_extraction_llm = primary_llm.with_structured_output(ExtractionResult)

# Build the chains
extraction_chain = extraction_prompt | structured_extraction_llm
sql_chain = sql_generation_prompt | primary_llm | StrOutputParser()
# recommendation_chain = recommendation_prompt | creative_llm
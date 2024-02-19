from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class AnswerOutputParser(BaseOutputParser):
    def parse(self, text: str):
        """Parses the output of an LLM call, extracting steps and answer."""
        parsed_text = text.strip().split("answer =")
        if len(parsed_text) == 2:
            steps, answer = parsed_text
            return steps.strip(), answer.strip().lower()  # Ensure lowercase answer
        else:
            raise ValueError("Invalid output format: answer = not found or multiple instances found.")

chat_model = ChatOpenAI(openai_api_key=api_key)

template = """You are a helpful assistant that solves math problems and shows your work.
             Please provide a clear and detailed explanation of each step,
             and then return the answer in all lowercase letters,
             following the format: answer = <answer here>.
             """
human_template = "{problem}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

problem = "2x^2 - 5x + 3 = 0"
messages = chat_prompt.format_messages(problem=problem)

try:
    result = chat_model.predict_messages(messages)
    parsed = AnswerOutputParser().parse(result.content)
    steps, answer = parsed

    print("Steps:", steps)
    print("Answer:", answer)
except ValueError as e:
    print("Error parsing output:", e)
except Exception as e:
    print("An unexpected error occurred:", e)

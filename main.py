import os

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course")
    print(os.environ.get("OLLAMA_HOST"))

    information = '''Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is an industrialist, technologist, and former public official who is the chief executive officer (CEO) and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and became the only trillionaire (in terms of US dollars) in June 2026; as of October 2026, Forbes estimates his net worth to be $1.1 trillion.

    Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal. Musk also became an American citizen in 2002.
    '''
    summary_template = '''
    Given this {information} about a person, I want you to 
    1) Summarise their monthly take home in INR
    2) Two interesting facts about them
    '''

    summary_prompt_template = PromptTemplate(
        input_variables = ['information'], template = summary_template
    )

    llm = ChatOllama(temperature = 0, model = "gemma3:270m")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={'information':information})

    print('\n\nResponse from LLM:\n')
    print(response.content)


if __name__ == "__main__":
    main()

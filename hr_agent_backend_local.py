# load core modules
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# load agents and tools modules
import pandas as pd
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain import LLMMathChain

#Using ChromaDB as a vector store for the embeddigns
from langchain.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Load all the .txt files from docs directory
docs = TextLoader("./docs/hr_policy.txt").load()

#Split text into tokens
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

#Turn the text into embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

#Store the embeddings into chromadb directory
docsearch = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chromadb")

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

print("Checkpoint E")

# initialize vectorstore retriever object
timekeeping_policy = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
)

df = pd.read_csv("employee_data.csv")  # load employee_data.csv as dataframe
python = PythonAstREPLTool(
    locals={"df": df}
)  # set access of python_repl tool to the dataframe

# create calculator tool
calculator = LLMMathChain(llm=llm)

# create variables for f strings embedded in the prompts
user = "Alexander Verdad"  # set user
df_columns = df.columns.to_list()  # print column names of df

# prep the (tk policy) vectordb retriever, the python_repl(with df access) and langchain calculator as tools for the agent
tools = [
    Tool(
        name="Timekeeping Policies",
        func=timekeeping_policy.run,
        description="""
        Useful for when you need to answer questions about employee timekeeping policies.

        <user>: What is the policy on unused vacation leave?
        <assistant>: I need to check the timekeeping policies to answer this question.
        <assistant>: Action: Timekeeping Policies
        <assistant>: Action Input: Vacation Leave Policy - Unused Leave
        ...
        """,
    ),
    Tool(
        name="Employee Data",
        func=python.run,
        description=f"""
        Useful for when you need to answer questions about employee data stored in pandas dataframe 'df'. 
        Run python pandas operations on 'df' to help you get the right answer.
        'df' has the following columns: {df_columns}
        
        <user>: How many Sick Leave do I have left?
        <assistant>: df[df['name'] == '{user}']['sick_leave']
        <assistant>: You have n sick leaves left.              
        """,
    ),
    Tool(
        name="Calculator",
        func=calculator.run,
        description=f"""
        Useful when you need to do math operations or arithmetic.
        """,
    ),
]

# change the value of the prefix argument in the initialize_agent function. This will overwrite the default prompt template of the zero shot agent type
agent_kwargs = {
    "prefix": f"You are friendly HR assistant. You are tasked to assist the current user: {user} on questions related to HR. You have access to the following tools:"
}

# initialize the LLM agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs=agent_kwargs,
)


# define q and a function for frontend
def get_response(user_input):
    response = agent.run(user_input)
    return response

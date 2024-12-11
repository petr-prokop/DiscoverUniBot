from typing import Annotated, Dict
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import langchain
import wikipedia
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, JsonOutputToolsParser
from pathlib import Path
from langchain_chroma.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


langchain.debug=False

class VectorStoreManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.persist_directory = "data/chroma_db"
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vectorstore = self._load_or_create_vectorstore()
            self._initialized = True

    def _load_or_create_vectorstore(self) -> Chroma:
        """Load existing vectorstore or create new one if not exists"""
        if self._database_exists():
            print("Loading existing vector database...")
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("Creating new vector database...")
            return self._create_new_vectorstore()

    def _database_exists(self) -> bool:
        """Check if database exists at persistence directory"""
        db_path = Path(self.persist_directory)
        return db_path.exists() and len(list(db_path.glob('*'))) > 0

    def _create_new_vectorstore(self) -> Chroma:
        """Create new vectorstore from documents"""
        # Ensure directory exists
        Path(self.persist_directory).parent.mkdir(parents=True, exist_ok=True)
        
        # Load and process documents
        loader = TextLoader("data/faq.html")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)
        
        # Create and persist vectorstore
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return vectorstore

    def similarity_search(self, query: str, k: int = 3):
        """Perform similarity search"""
        return self.vectorstore.similarity_search(query=query, k=k)

@tool
def get_study_info(question: str) -> str:
    """Search FAQ database for relevant information."""
    vector_manager = VectorStoreManager()
    docs = vector_manager.similarity_search(question)
    
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"Result {i}:\n{doc.page_content}")
    
    return "\n\n".join(results) if results else "No relevant information found."

# Replace original vector store initialization with single line
vector_manager = VectorStoreManager()

@tool
def get_wikipedia_content(search_query: str):
    """Get top 3 results from Wikipedia for the search query."""
    try:
        page_titles = wikipedia.search(search_query)
        summaries = []
        for page_title in page_titles[:3]:
            try:
                wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
                summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
            except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
                continue
        return "\n\n".join(summaries) if summaries else "No relevant Wikipedia information found."
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

tools = [get_study_info, get_wikipedia_content]
tools_dict = {'get_study_info': get_study_info, 'get_wikipedia_content': get_wikipedia_content}

local_llm_type = 'llama3.3'
local_llm_type = 'llama3.1:8b-instruct-fp16'

llm = ChatOllama(model=local_llm_type, temperature=0, verbose=True)
llm_with_tools = llm.bind_tools(tools)
llm_json_format = ChatOllama(model=local_llm_type, format='json', temperature=0)


class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    # for message in state['messages']:
    #     print(f"{type(message)}: {message.content}")
    return {"messages": [llm.invoke(state["messages"])]}

def router(state: State):
    # Create intent analysis chain
    intent_prompt = ChatPromptTemplate.from_template("""
    You are a university chatbot assistant that analyzes user questions.
    Determine the type of query from the following categories:
    - study_info: Questions about university studies, programs, or admissions
    - get_wikipedia_content: Questions about encyklopedia information
    - chatbot: Other casual conversation
    
    Return the classification as JSON with a single key 'intent'.
    
    User message: {message}
    """)
    
    chain = intent_prompt | llm_json_format | JsonOutputParser()
    
    try:
        message = state["messages"][-1].content if state["messages"] else ""
        result = chain.invoke({"message": message})
        intent = result.get("intent", "chatbot")
        return {"messages": [], "next": intent}
    except Exception as e:
        print(f"Error: {e}")
        return {"messages": [], "next": "chatbot"}

def get_external_info(state: State):
    """
    Use LLM with binded tools to get decision about function call.
    Make a function call and append the result to the state.
    """
    results = []
    chain = llm_with_tools
    llm_with_tools_response = chain.invoke(state["messages"])
    results.append(llm_with_tools_response)
    
    if llm_with_tools_response:
        if tools_condition(llm_with_tools_response.tool_calls):
            
            for tool_call in llm_with_tools_response.tool_calls:
                print(f"Calling: {tool_call}")
                tool_response = tools_dict[tool_call['name']].invoke(tool_call['args'])
                results.append({
                    "role": "tool",
                    "content": f"Calling tool: {tool_call['name']} with args: {tool_call['args']}",
                    "tool_call_id": tool_call['id']
                })
                results.append({
                    "role": "tool",
                    "content": f"Tool response: {tool_response}",
                    "tool_call_id": tool_call['id']
                })

    return {"messages": results}

graph_builder.add_node("router", router)
graph_builder.set_entry_point("router")
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", get_external_info)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

def route_based_on_intent(state_dict: Dict) -> str:
    next_state = state_dict.get('next', 'chatbot')
    print(f"Routing state: {state_dict}")
    if next_state == 'chatbot':
        return 'chatbot'
    else:
        return 'tools'

# Add conditional edges with explicit routing
graph_builder.add_conditional_edges(
    "router",
    route_based_on_intent,
    {
        "chatbot": "chatbot",
        "tools": "tools",
    }
)

memory = MemorySaver()
app = graph_builder.compile(
    checkpointer=memory,
)

app.get_graph().draw_mermaid_png(output_file_path="data/conversational_agent_tools_info_retrieval.png")
    
def chat_loop():
    print("🤖 University Chat Assistant (type 'exit' to quit)")
    app.invoke({"messages": [('system', """Welcome to the University Chat Assistant!
You are a friendly university chatbot assistant. Your personality traits are:
- Helpful and supportive
- Light and occasionally humorous
- Engaging and conversational
- Empathetic to student concerns

You have access to university FAQ database and weather information.
When needed, you will use these external sources to provide accurate information.
Always maintain a friendly tone while being informative.               
""")]}, config={"configurable": {"thread_id": "1"}})

    while True:
        msg = input("\nYou: ").strip()
        if msg.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        try:
            for chunk in app.stream(
                {"messages": [("human", msg)]}, 
                config={"configurable": {"thread_id": "1"}}, 
                stream_mode="values"
            ):
                chunk["messages"][-1].pretty_print()
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_loop()
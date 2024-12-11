from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, Graph
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
import wikipedia
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

# State Models
class ConversationState(BaseModel):
    messages: List[str] = Field(default_factory=list)
    current_topic: Optional[str] = None
    rag_context: Optional[Dict] = None
    email_data: Optional[Dict] = None

class IntentAnalyzer:
    def __init__(self, llm):
        self.llm = llm
        self.intent_prompt = PromptTemplate(
            template="""You are a university chatbot assistant that analyzes user questions.
            Determine the type of query from the following categories:
            - study_info: Questions about university studies, programs, or admissions
            - wiki: General knowledge questions
            - email: Requests to send information via email
            - general: Other casual conversation
            
            Return the classification as JSON with a single key 'intent'.
            
            User message: {message}
            """,
            input_variables=["message"]
        )
        self.parser = JsonOutputParser()

def analyze_intent(state: ConversationState) -> Dict:
    """Analyzes user intent and routes to appropriate handler"""
    analyzer = IntentAnalyzer(llm)
    current_message = state.messages[-1] if state.messages else ""
    
    prompt = analyzer.intent_prompt.format(message=current_message)
    response = analyzer.llm.invoke(prompt)
    intent = analyzer.parser.parse(response.content)
    # Return state updates directly
    return {
        "messages": state.messages,
        "current_topic": intent['intent'],
        "rag_context": state.rag_context,
        "email_data": state.email_data,
        "__next": intent['intent']  # Use special key for routing
    }

class UniversityRAG:
    def __init__(self, llm):
        self.llm = llm
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.setup_rag()

    def setup_rag(self):
        # Load documents
        loader = TextLoader("data/faq.html")
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def get_rag_chain(self):
        # Create prompt for combining documents
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question about university studies using the provided context.
        If you cannot find relevant information, provide a general response.
        
        Context: {context}
        Question: {input}
        
        Answer:""")

        # Create chain to combine documents
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create retrieval chain
        return create_retrieval_chain(self.retriever, document_chain)

def handle_study_info(state: ConversationState) -> Dict:
    """Handles university-related queries using RAG"""
    rag = UniversityRAG(llm)
    chain = rag.get_rag_chain()
    
    # Get user question from state
    question = state.messages[-1]
    
    # Run RAG chain
    response = chain.invoke({"input": question})
    answer = response["answer"]
    
    # Update state
    state.messages.append(answer)
    state.rag_context = {"query_processed": True, "documents_retrieved": True}
    
    return {
        "messages": state.messages,
        "current_topic": state.current_topic,
        "rag_context": state.rag_context,
        "email_data": state.email_data,
        "__next": "output",
        "response": answer
    }

def handle_email(state: ConversationState) -> Dict:
    """Handles email requests"""
    response = "I'll help you send information via email. Please provide your email address."
    state.messages.append(response)
    
    return {
        "messages": state.messages,
        "current_topic": state.current_topic,
        "rag_context": state.rag_context,
        "email_data": state.email_data,
        "__next": "output",
        "response": response
    }

class WikipediaChain:
    def __init__(self, llm):
        self.llm = llm
        self.setup_chain()
    
    def setup_chain(self):
        self.prompt = ChatPromptTemplate.from_template("""
        Based on the Wikipedia information provided, answer the user's question.
        If no relevant information is found, provide a general response.
        
        Wikipedia Information: {wiki_content}
        Question: {question}
        
        Answer:""")
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def search_wikipedia(self, query: str) -> str:
        try:
            page_titles = wikipedia.search(query)
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

def handle_wikipedia(state: ConversationState) -> Dict:
    """Handles general knowledge queries using Wikipedia"""
    wiki_chain = WikipediaChain(llm)
    
    # Get user question
    question = state.messages[-1]
    
    # Search Wikipedia
    wiki_content = wiki_chain.search_wikipedia(question)
    
    # Generate response using LLM
    response = wiki_chain.chain.invoke({
        "wiki_content": wiki_content,
        "question": question
    })
    answer = response["text"]
    
    # Update state
    state.messages.append(answer)
    
    return {
        "messages": state.messages,
        "current_topic": state.current_topic,
        "rag_context": state.rag_context,
        "email_data": state.email_data,
        "__next": "output",
        "response": answer
    }

class GeneralConversationHandler:
    def __init__(self, llm):
        self.llm = llm
        self.memory = ConversationBufferMemory(return_messages=True)
        self.setup_chain()
    
    def setup_chain(self):
        self.prompt = ChatPromptTemplate.from_template("""
        You are a friendly university chatbot assistant. Your personality traits are:
        - Helpful and supportive
        - Light and occasionally humorous
        - Engaging and conversational
        - Empathetic to student concerns
        
        Previous conversation:
        {history}
        
        Current message: {message}
        
        Respond in a natural, friendly way while maintaining professionalism.
        """)
        
        self.chain = self.LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

def handle_general_conversation(state: ConversationState) -> Dict:
    """Handles casual conversation with friendly persona"""
    # handler = GeneralConversationHandler(llm)
    template = """
You are a friendly university chatbot assistant. Your personality traits are:
- Helpful and supportive
- Light and occasionally humorous
- Engaging and conversational
- Empathetic to student concerns
"""

    instructions = """Previous conversation:
{history}

Current message: {message}

Respond in a natural, friendly way while maintaining professionalism.
        """
    # Get conversation history
    history = "\n".join(state.messages[:-1])
    current_message = state.messages[-1]

    messages = [
        SystemMessage(
            content=template
        ),
        HumanMessage(
            content=instructions.format(history=history, message=current_message)
        )
    ]
    response = llm_no_json.invoke(messages)
    
    # Update state
    state.messages.append(response.content)
    
    return {
        "messages": state.messages,
        "current_topic": "general",
        "rag_context": state.rag_context,
        "email_data": state.email_data,
        "__next": "output",
        "response": response.content
    }

def output_response(state: ConversationState) -> Dict:
    """Final node to format output"""
    ## print all from state.messages
    # for message in state.messages:
    #     print(message)
    return {
        "messages": state.messages,
        "current_topic": state.current_topic,
        "rag_context": state.rag_context,
        "email_data": state.email_data,
        "response": state.messages[-1] if state.messages else "No response generated."
    }

# Graph Construction
def create_conversation_graph() -> Graph:
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("intent_analysis", analyze_intent)
    workflow.add_node("study_handler", handle_study_info)
    workflow.add_node("email_handler", handle_email)
    workflow.add_node("wiki_handler", handle_wikipedia)
    workflow.add_node("general_handler", handle_general_conversation)
    workflow.add_node("output", output_response)
    
    # Define edge routing function
    def route_based_on_intent(state_dict: Dict) -> str:
        # Print for debugging
        print(f"Routing state: {state_dict}")
         # Convert state to dict if needed
        if isinstance(state_dict, ConversationState):
            state_dict = dict(state_dict)
        # Get intent with fallback
        intent = state_dict.get("current_topic") or "general"
        return intent
    
    # Add conditional edges with explicit routing
    workflow.add_conditional_edges(
        "intent_analysis",
        route_based_on_intent,
        {
            "study_info": "study_handler",
            "email": "email_handler",
            "wiki": "wiki_handler",
            "general": "general_handler"
        }
    )
    
    # Ensure all handlers route to output
    for handler in ["study_handler", "email_handler", "wiki_handler", "general_handler"]:
        workflow.add_edge(handler, "output")
    
    workflow.set_entry_point("intent_analysis")
    return workflow.compile()

import langchain
langchain.debug = False
# LLM
local_llm = 'llama3.1:8b-instruct-fp16'
# local_llm = 'llama3.3'

llm = ChatOllama(model=local_llm, format="json", temperature=0)
llm_no_json = ChatOllama(model=local_llm, temperature=0)

# Create initial state
state = ConversationState(
    messages=[],
    current_topic=None,
    rag_context={},
    email_data={}
)

# Create graph
graph = create_conversation_graph()

print("University Assistant Bot (type 'quit' or 'bye' to exit)")

while True:
    try:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check exit conditions
        if user_input.lower() in ['quit', 'bye']:
            print("\nAssistant: Goodbye! Have a great day!")
            break
            
        # Update state with user input
        state.messages.append(user_input)
        
        # Process through graph
        result = graph.invoke(state)
        response = result["messages"][-1]
        print(f"\nAssistant: {response}")
        
        # Clear message history to prevent state bloat
        state.messages = state.messages[-2:]
        
    except Exception as e:
        # propagate exception
        print(f"\nError: {str(e)}")
        print("Assistant: I encountered an error. Please try again.")

        raise e
            
            
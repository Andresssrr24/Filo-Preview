'''V1, deprecated'''
'''RAG system made to answer questions about The Republic with Classic ReAct Agent (Thought -> Action -> Observation) approach '''
#from langchain_chroma import Chroma
#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.schema import Document
#from llama_index.core.node_parser import SentenceSplitter
#from llama_index.readers.file import PDFReader # Pypdf in backend
#from langchain_groq import ChatGroq
#from langchain.prompts import PromptTemplate, ChatPromptTemplate
#from langchain.retrievers.multi_query import MultiQueryRetriever
#from langchain.agents import AgentExecutor, create_react_agent
#from langchain_core.tools import StructuredTool
#from langchain_tavily import TavilySearch
'''from app.rag.pipelines.vecstores import VecStore'''
#import os
#from dotenv import load_dotenv
#from datetime import datetime
#import app.settings as settings

#load_dotenv()

#retriever_tools_settings = settings.RETRIEVER_TOOLS_SETTINGS
'''tools_settings = settings.TOOLS_SETTINGS
templates = settings.PROMPT_TEMPLATES
models_settings = settings.MODELS_SETTINGS'''
#context_setting = settings.CONTEXT_SETTINGS
#states = settings.STATES

'''class RetrieverTools:
    """Retriever tools for llm"""
    def __init__(self, vecstore, query_generator_llm, tavily_api_key=os.getenv("TAVILY_API_KEY")):
        self.base_retriever = vecstore.as_retriever(search_type=retriever_tools_settings["base_retriever_search_type"], search_kwargs={"k": retriever_tools_settings["base_retriever_k"]})
        self.multi_q_retriever = MultiQueryRetriever.from_llm(include_original=True, retriever=vecstore.as_retriever(search_type=retriever_tools_settings["multi_query_search_type"], search_kwargs={"k": retriever_tools_settings["multi_query_k"]}), llm=query_generator_llm)
        self.web_search = TavilySearch(api_key=tavily_api_key, max_results=retriever_tools_settings["web_search_k"], search_depth=retriever_tools_settings["search_depth"])

    def _format_output(self, docs: list) -> str:
        form_docs = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get('page_label', 'N/A')
            source = doc.metadata.get('file_name', 'N/A')
            score = doc.metadata.get('score', 'N/A')

            form_docs.append(
                f"Document {i}:\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Score: {score}\n"
                f"Content: {doc.page_content}"
            )

        return "\n\n".join(form_docs)

    # Base retriever tool
    def base_retriever_tool(self, query: str) -> str:
        """Basic embedding search using mmr with LangChain built-in retriever """
        try:
            docs = self.base_retriever.invoke(query)

            if not docs:
                return "No relevant docs found."
            
            return self._format_output(docs)
        
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"

    # Multi-Query retriever tool
    def multi_query_retriever_tool(self, query: str) -> str: # TODO: optimize main llm token usage
        """Retrieve documents by generating multiple synthetic queries with an llm """
        try:
            docs = self.multi_q_retriever.invoke(query)
            
            if not docs:
                print("No docs found, changing to base_retriever")
                return self.base_retriever_tool(query)
            
            return self._format_output(docs)
        except:
            print("Couldn't use multi-query retriever, changing to base_retriever")
            return self.base_retriever_tool(query)
        
    def web_search_tool(self, query: str) -> str: 
        """Search the web for information related to the query"""
        try:
            results = self.web_search.invoke(query)

            if not results:
                return f"No web results found for {query}."

            """formatted = "\n".join(
                f"- {r['title']}: {r['content']} (source: {r['url']})"
                for r in results
            )"""
            return f"Web search results for '{query}':\n{results}" 
        
        except Exception as e:
            return f"Error during web search: {str(e)}"'''

'''class ContextBuilder:
    def __init__(self, chat_vecstore, models):
        self.chat_vecstore = chat_vecstore
        self.decision = states["context_decision"][0]
        self.models = models

    def decide_context(self, query: str) -> str:
        """Decide wich type of context is needed"""
        msg = query.lower().strip()

        if len(msg) < 5 and any(word in msg for word in context_setting["short_context"]):
            self.decision = states["context_decision"][1]
            return self.decision
        
        if any(kw in msg for kw in context_setting["long_context"]):
            self.decision = states["context_decision"][2]
            return self.decision
        
        if msg in context_setting["no_context"]:
            self.decision = states["context_decision"][0]
            return self.decision
        
        # If it is not clear, use small llm to decide
        self.decision = self.models['small_llm'].invoke(
            templates["decide_context_prompt"].format(query=query)
        ).content.strip()

        return self.decision
    
    def build_context(self) -> str:
        """Use context methods depending of type of context"""
        if self.decision == states["context_decision"][0]:
            return ""
        
        elif self.decision == states["context_decision"][1]:
            self.chat_vecstore.get_buffer_memory
            
        elif self.decision == states["context_decision"][2]:
            return "\n".join([doc.page_content for doc in self.chat_vecstore.get_k_messages()])'''
        
'''class AgentMaker:
    def __init__(self, retriever_tools: RetrieverTools, models: dict):
        self.tools = []
        self.router_chain = None
        self.retriever_tools = retriever_tools
        self.models = models

    # Create agent tools
    def agent_tools(self) -> list:
        self.tools = [
            StructuredTool.from_function(
                name=str(tools_settings["names"][0]),
                func=self.retriever_tools.base_retriever_tool,
                description=str(tools_settings["descriptions"][0])
            ),
            StructuredTool.from_function(
                name=str(tools_settings["names"][1]),
                func=self.retriever_tools.multi_query_retriever_tool,
                description=str(tools_settings["descriptions"][1])
            ),
            StructuredTool.from_function(
                name=str(tools_settings["names"][2]),
                func=self.retriever_tools.web_search_tool,
                description=str(tools_settings["descriptions"][2])
            ),
        ]
        return self.tools

    # Create agent
    def agent(self) -> AgentExecutor:
        prompt = PromptTemplate.from_template(
            templates["main_prompt"]
        )

        # react agent
        agent_builder = create_react_agent(
            llm=self.models["main_llm"],
            tools=self.tools,
            prompt=prompt
        )

        # executor with additional params
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent_builder,
            tools=self.tools,
            verbose=True,
            max_iterations=3,
            #max_execution_time=300,
            handle_parsing_errors=True,
        )

        return agent_executor
    
    # TODO: Up to this point starts base_agent
    def first_response_chain(self):
        small_llm_prompt = ChatPromptTemplate(
            [
                ("system", templates["small_llm_prompt"]),
                ("human", "{question}")
            ]
        )

        self.router_chain = small_llm_prompt | self.models['small_llm']
        return self.router_chain

    def router(self, query: str, executor, context_builder) -> str:
        """"Decide if answer with small or main llm"""
        action_decision = self.router_chain.invoke({"question": query}).content.strip()
        print(action_decision)

        if action_decision == states['model_choice'][0]:
            return self.models['small_llm'].invoke(query).content
        
        decision_context = context_builder.decide_context(query)
        print(f"Decided context: {decision_context}")
        # Context-aware query if needed
        if states['context_decision'][0] not in decision_context:
            print("Building context") 
            context = context_builder.build_context()
            query = f"Context:\n{context}\n\nQuestion: {query}"

        if action_decision == states['model_choice'][1]:
            rsp = executor.invoke({"question": query})
            return rsp["output"]

        elif action_decision == states['model_choice'][2]:
            return self.models['creative_llm'].invoke(query).content

        else:
            return "I'm sorry, I couldn't understand your request."'''
        
# Initialize LLMs
'''def llms() -> dict:
    try:
        models = {
            "main_llm": ChatGroq(
                api_key=os.getenv('GROQ_API_KEY'),
                model=models_settings["main_llm"]["model"], 
                max_retries=models_settings["main_llm"]["max_retries"],
                temperature=models_settings["main_llm"]["temperature"],
            ),

            "small_llm": ChatGroq(
                api_key=os.getenv('GROQ_API_KEY'),
                model=models_settings["small_llm"]["model"],
                max_retries=models_settings["small_llm"]["max_retries"],
                temperature=models_settings["small_llm"]["temperature"],
            ),

            "query_generator": ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=models_settings["query_generator"]["model"],
                max_retries=models_settings["query_generator"]["max_retries"],
                temperature=models_settings["query_generator"]["temperature"],
                max_tokens=models_settings["query_generator"]["max_tokens"],
            ),

            "creative_llm": ChatGroq(
                api_key=os.getenv('GROQ_API_KEY'),
                model=models_settings["creative_llm"]["model"],
                temperature=models_settings["creative_llm"]["temperature"],
                max_retries=models_settings["creative_llm"]["max_retries"],
            ),
        }
    except Exception as e:
        print(f"Error initializing LLMs: {e}")
        raise("Error initializing LLMs.")
        
    return models'''
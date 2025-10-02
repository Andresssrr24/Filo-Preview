from langchain_core.tools import StructuredTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from tools.retrieval.retrievers import RetrieverTools
from app import settings
import logging

logger = logging.getLogger(__name__)

tools_settings = settings.TOOLS_SETTINGS
prompt_templates = settings.PROMPT_TEMPLATES
states = settings.STATES

class FactualAgent:
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
            prompt_templates["main_prompt"]
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
                ("system", prompt_templates["small_llm_prompt"]),
                ("human", "{question}")
            ]
        )

        self.router_chain = small_llm_prompt | self.models['small_llm']
        return self.router_chain

    def router(self, query: str, executor, context_builder) -> str:
        """"Decide if answer with small or main llm"""
        action_decision = self.router_chain.invoke({"question": query}).content.strip()
        logger.info(f"Action decision: {action_decision}")

        if action_decision == states['model_choice'][0]:
            return self.models['small_llm'].invoke(query).content
        
        decision_context = context_builder.decide_context(query)
        logger.info(f"Decided context: {decision_context}")
        # Context-aware query if needed
        if states['context_decision'][0] not in decision_context:
            logger.info("Building context") 
            context = context_builder.build_context()
            query = f"Context:\n{context}\n\nQuestion: {query}"

        if action_decision == states['model_choice'][1]:
            rsp = executor.invoke({"question": query})
            return rsp["output"]

        elif action_decision == states['model_choice'][2]:
            return self.models['creative_llm'].invoke(query).content

        else:
            return "I'm sorry, I couldn't understand your request."
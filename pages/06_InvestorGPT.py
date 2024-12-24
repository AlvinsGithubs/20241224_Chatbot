from langchain.schema import SystemMessage
import streamlit as st
import os
import yfinance as yf  # yfinance 라이브러리 추가
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o")

# yfinance를 사용하므로 Alpha Vantage API 키는 필요 없음


class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(
        description="회사의 이름을 입력하세요. 예: 삼성전자"
    )


class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = """
    이 도구는 회사 이름으로부터 주식 심볼을 찾는 데 사용됩니다.
    예를 들어, "삼성전자"를 입력하면 "005930.KS"와 같은 심볼을 반환합니다.
    """
    args_schema: Type[
        StockMarketSymbolSearchToolArgsSchema
    ] = StockMarketSymbolSearchToolArgsSchema

    def _run(self, query):
        # yfinance의 Ticker 객체를 사용하여 심볼 찾기
        ticker = yf.Ticker(query)
        info = ticker.info
        symbol = info.get('symbol', None)
        if symbol:
            return symbol
        else:
            return "해당 회사의 주식 심볼을 찾을 수 없습니다."


class CompanyOverviewArgsSchema(BaseModel):
    symbol: str = Field(
        description="회사 주식 심볼을 입력하세요. 예: 005930.KS"
    )


class CompanyOverviewTool(BaseTool):
    name = "CompanyOverview"
    description = """
    이 도구는 회사의 재무 개요를 가져오는 데 사용됩니다.
    주식 심볼을 입력하세요.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info


class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatement"
    description = """
    이 도구는 회사의 손익계산서를 가져오는 데 사용됩니다.
    주식 심볼을 입력하세요.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        ticker = yf.Ticker(symbol)
        income_stmt = ticker.financials
        return income_stmt.to_dict()


class CompanyStockPerformanceTool(BaseTool):
    name = "CompanyStockPerformance"
    description = """
    이 도구는 회사 주식의 주간 성과를 가져오는 데 사용됩니다.
    주식 심볼을 입력하세요.
    """
    args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

    def _run(self, symbol):
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="1y", interval="1wk")
        return history.to_dict()


agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        CompanyIncomeStatementTool(),
        CompanyStockPerformanceTool(),
        StockMarketSymbolSearchTool(),
        CompanyOverviewTool(),
    ],
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
            당신은 한국어로 대화해야 합니다.

            당신은 헤지 펀드 매니저입니다.
            
            당신은 회사를 평가하고 해당 주식이 매수할 만한지 여부에 대한 의견과 그 이유를 제공합니다.
            
            주식의 성과, 회사 개요 및 손익계산서를 고려하세요.
            
            판단에 있어서 단호하게 행동하고 주식을 추천하거나 사용자에게 주식을 피할 것을 조언하세요.
        """
        )
    },
)

st.set_page_config(
    page_title="InvestorGPT",
    page_icon="💼",
)

st.markdown(
    """
    # InvestorGPT
            
    Welcome to InvestorGPT.
            
    관심 있는 회사의 이름을 작성하면 저희 에이전트가 연구를 수행해 드립니다.
"""
)

company = st.text_input("관심 있는 회사의 이름을 작성하세요.")

if company:
    result = agent.invoke(company)
    st.write(result["output"].replace("$", "\$"))



#------------ver1.0
# from langchain.schema import SystemMessage
# import streamlit as st
# import os
# import requests
# from typing import Type
# from langchain.chat_models import ChatOpenAI
# from langchain.tools import BaseTool
# from pydantic import BaseModel, Field
# from langchain.agents import initialize_agent, AgentType
# from langchain.utilities import DuckDuckGoSearchAPIWrapper
# from dotenv import load_dotenv

# # 환경 변수 로드
# load_dotenv()

# llm = ChatOpenAI(temperature=0.1, model_name="gpt-4o")

# alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")


# class StockMarketSymbolSearchToolArgsSchema(BaseModel):
#     query: str = Field(
#         description="The query you will search for.Example query: Stock Market Symbol for Apple Company"
#     )


# class StockMarketSymbolSearchTool(BaseTool):
#     name = "StockMarketSymbolSearchTool"
#     description = """
#     Use this tool to find the stock market symbol for a company.
#     It takes a query as an argument.
    
#     """
#     args_schema: Type[
#         StockMarketSymbolSearchToolArgsSchema
#     ] = StockMarketSymbolSearchToolArgsSchema

#     def _run(self, query):
#         ddg = DuckDuckGoSearchAPIWrapper()
#         return ddg.run(query)


# class CompanyOverviewArgsSchema(BaseModel):
#     symbol: str = Field(
#         description="Stock symbol of the company.Example: AAPL,TSLA",
#     )


# class CompanyOverviewTool(BaseTool):
#     name = "CompanyOverview"
#     description = """
#     Use this to get an overview of the financials of the company.
#     You should enter a stock symbol.
#     """
#     args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

#     def _run(self, symbol):
#         r = requests.get(
#             f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}"
#         )
#         return r.json()


# class CompanyIncomeStatementTool(BaseTool):
#     name = "CompanyIncomeStatement"
#     description = """
#     Use this to get the income statement of a company.
#     You should enter a stock symbol.
#     """
#     args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

#     def _run(self, symbol):
#         r = requests.get(
#             f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}"
#         )
#         return r.json()["annualReports"]


# class CompanyStockPerformanceTool(BaseTool):
#     name = "CompanyStockPerformance"
#     description = """
#     Use this to get the weekly performance of a company stock.
#     You should enter a stock symbol.
#     """
#     args_schema: Type[CompanyOverviewArgsSchema] = CompanyOverviewArgsSchema

#     def _run(self, symbol):
#         r = requests.get(
#             f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}"
#         )
#         response = r.json()
#         return list(response["Weekly Time Series"].items())[:200]


# agent = initialize_agent(
#     llm=llm,
#     verbose=True,
#     agent=AgentType.OPENAI_FUNCTIONS,
#     handle_parsing_errors=True,
#     tools=[
#         CompanyIncomeStatementTool(),
#         CompanyStockPerformanceTool(),
#         StockMarketSymbolSearchTool(),
#         CompanyOverviewTool(),
#     ],
#     agent_kwargs={
#         "system_message": SystemMessage(
#             content="""
#             When you speak, You should speak in Korean.

#             You are a hedge fund manager.
            
#             You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
            
#             Consider the performance of a stock, the company overview and the income statement.
            
#             Be assertive in your judgement and recommend the stock or advise the user against it.
#         """
#         )
#     },
# )

# st.set_page_config(
#     page_title="InvestorGPT",
#     page_icon="💼",
# )

# st.markdown(
#     """
#     # InvestorGPT
            
#     Welcome to InvestorGPT.
            
#     Write down the name of a company and our Agent will do the research for you.
# """
# )

# company = st.text_input("Write the name of the company you are interested on.")

# if company:
#     result = agent.invoke(company)
#     st.write(result["output"].replace("$", "\$"))
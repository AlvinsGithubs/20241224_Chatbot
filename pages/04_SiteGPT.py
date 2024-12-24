# pages/04_SiteGPT.py

from dotenv import load_dotenv
import os
import streamlit as st
import requests
from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup  # BeautifulSoup 설치 필요

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("`OPENAI_API_KEY`가 설정되지 않았습니다. `.env` 파일을 확인하세요.")
    st.stop()

# OpenAI LLM 초기화
llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=openai_api_key,  # API 키 명시적으로 전달
)

# ChatPromptTemplate 정의
answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                          
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                          
    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question: {question}
"""
)

# get_answers 함수 정의
def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata.get("source", "Unknown"),
                "date": doc.metadata.get("lastmod", "Unknown"),
            }
            for doc in docs
        ],
    }

# choose_prompt 정의
choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

# choose_answer 함수 정의
def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

# 페이지 파싱 함수 정의
def parse_page(content):
    soup = BeautifulSoup(content, "html.parser")
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    # 추가적으로 필요한 요소가 있으면 제거
    return (
        soup.get_text(separator=" ", strip=True)
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )

# 사이트맵 로드 함수 정의
@st.cache_data(show_spinner="Loading sitemap...")
def load_website_sitemap(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    if not docs:
        raise ValueError("사이트맵에서 문서를 로드하지 못했습니다.")
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))
    return vector_store.as_retriever()

# 개별 페이지 로드 함수 정의
@st.cache_data(show_spinner="Loading page...")
def load_individual_page(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
                      " Chrome/58.0.3029.110 Safari/537.3"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"URL을 로드하는 중 오류가 발생했습니다: {e}")

    # HTML 콘텐츠 파싱
    parsed_content = parse_page(response.text)
    if not parsed_content.strip():
        raise ValueError("파싱된 문서가 비어 있습니다.")

    # Document 객체 생성
    from langchain.schema import Document
    doc = Document(page_content=parsed_content, metadata={"source": url, "lastmod": "Unknown"})
    
    # 텍스트 분할
    docs = splitter.split_documents([doc])
    if not docs:
        raise ValueError("문서 분할에 실패했습니다.")
    
    # 디버깅: 첫 번째 문서의 내용을 출력
    st.write("Loaded Content (첫 번째 문서):")
    st.write(docs[0].page_content[:500])  # 처음 500자만 표시

    # 벡터 스토어 생성
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # 디버깅: 첫 번째 벡터 출력
    if vector_store.index.ntotal > 0:
        vectors = vector_store.index.reconstruct_n(0, 1)  # 첫 번째 벡터
        st.write("First Vector:", vectors)
    else:
        st.warning("벡터 스토어에 벡터가 없습니다.")

    return vector_store.as_retriever()

# Streamlit 페이지 설정
st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

# Streamlit UI 구성
st.markdown(
    """
    # SiteGPT

    Ask questions about the content of a website.

    Start by writing the URL of the website on the sidebar.
    """
)

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com/sitemap.xml 또는 https://example.com/page",
    )
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")

if url:
    # 사이트맵 URL인지 개별 페이지 URL인지 판별
    if url.endswith(".xml"):
        st.sidebar.info("사이트맵 URL을 인식했습니다. 콘텐츠를 로드 중입니다...")
        try:
            retriever = load_website_sitemap(url)
        except Exception as e:
            st.sidebar.error(f"사이트맵을 로드하는 중 오류가 발생했습니다: {e}")
            retriever = None
    else:
        st.sidebar.info("개별 페이지 URL을 인식했습니다. 콘텐츠를 로드 중입니다...")
        try:
            retriever = load_individual_page(url)
        except Exception as e:
            st.sidebar.error(f"페이지를 로드하는 중 오류가 발생했습니다: {e}")
            retriever = None

    # 질문 입력
    query = st.text_input("Ask a question about the website's content.")

    if query and retriever:
        try:
            # Langchain 람다 체인 구성
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))
        except Exception as e:
            st.error(f"질문을 처리하는 중 오류가 발생했습니다: {e}")

else:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

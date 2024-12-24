from langchain.storage import LocalFileStore
import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from dotenv import load_dotenv
import re
import logging

# 환경 변수 로드
load_dotenv()

# 파일 이름을 안전하게 변환하는 함수
def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FFmpeg 설치 여부 확인 함수
def check_ffmpeg_installed():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

# FFmpeg가 설치되어 있는지 확인
if not check_ffmpeg_installed():
    st.error("FFmpeg가 시스템에 설치되어 있지 않거나, PATH에 포함되지 않았습니다. FFmpeg를 설치하고 다시 시도하세요.")
    st.stop()

# pydub에 FFmpeg 경로 직접 설정 (필요 시)
AudioSegment.converter = r"C:\Users\AlvinJang\Desktop\report_Maker\ffmpeg\bin\ffmpeg.exe"

# OpenAI API 키 로드
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API 키가 설정되지 않았습니다. `.env` 파일에 `OPENAI_API_KEY`를 추가하세요.")
    st.stop()

llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=openai_api_key,  # API 키 로드
)

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)

@st.cache_data()
def embed_file(file_path):
    try:
        file_name = os.path.basename(file_path)
        cache_dir = LocalFileStore(f"./.cache/embeddings/{file_name}")
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load_and_split(text_splitter=splitter)
        logger.info(f"Loaded {len(docs)} documents from {file_path}.")
        for i, doc in enumerate(docs):
            logger.info(f"Document {i+1}: {doc.page_content[:100]}...")
        embeddings = OpenAIEmbeddings()
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        retriever = vectorstore.as_retriever()
        return retriever
    except Exception as e:
        logger.error(f"Error embedding file {file_path}: {e}")
        st.error(f"임베딩 파일 생성 중 오류가 발생했습니다: {e}")
        raise e

@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    # 전사된 파일이 이미 존재하면 전사하지 않음
    if os.path.exists(destination):
        logger.info(f"Transcript file {destination} already exists. Skipping transcription.")
        return
    files = glob.glob(f"{chunk_folder}/*.mp3")  # MP3 파일로 유지
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a", encoding="utf-8") as text_file:
            try:
                transcript = openai.Audio.transcribe(
                    "whisper-1",
                    audio_file,
                )
                text_file.write(transcript["text"] + "\n")
                logger.info(f"Transcribed {file} successfully.")
            except Exception as e:
                logger.error(f"Error transcribing {file}: {e}")
                st.error(f"오디오 전사 중 오류가 발생했습니다: {e}")
    # 전사가 완료된 후 파일 확인
    if os.path.exists(destination):
        file_size = os.path.getsize(destination)
        if file_size > 0:
            logger.info(f"Transcript file {destination} created successfully with size {file_size} bytes.")
        else:
            logger.error(f"Transcript file {destination} is empty.")
            st.error("전사된 텍스트 파일이 비어 있습니다.")
    else:
        logger.error(f"Transcript file {destination} does not exist.")
        st.error("전사된 텍스트 파일이 생성되지 않았습니다.")

@st.cache_data()
def extract_audio_from_video(video_path):
    audio_path = os.path.splitext(video_path)[0] + ".mp3"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    try:
        subprocess.run(command, check=True)
        logger.info(f"Audio extracted successfully: {audio_path}")
        return audio_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg 실행 중 오류 발생: {e}")
        st.error(f"FFmpeg 실행 중 오류가 발생했습니다: {e}")
        raise e

@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if not os.path.exists(chunks_folder):
        os.makedirs(chunks_folder, exist_ok=True)
    try:
        track = AudioSegment.from_mp3(audio_path)
    except Exception as e:
        st.error(f"오디오 파일을 읽는 중 오류가 발생했습니다: {e}")
        logger.error(f"Error loading audio file {audio_path}: {e}")
        return
    chunk_len = chunk_size * 60 * 1000  # 분을 밀리초로 변환
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk_filename = f"chunk_{i}.mp3"
        try:
            chunk.export(
                os.path.join(chunks_folder, chunk_filename),
                format="mp3",
            )
            logger.info(f"Exported {chunk_filename}")
        except Exception as e:
            st.error(f"오디오 청크를 저장하는 중 오류가 발생했습니다: {e}")
            logger.error(f"Error exporting chunk {chunk_filename}: {e}")

# Streamlit 페이지 설정
st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

st.markdown(
    """
# MeetingGPT

Welcome to MeetingGPT! Upload a video file and I will provide you with a transcript, a summary, and a chatbot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov", "webm"],  # 'webm' 추가
    )

if video:
    chunks_folder = "./.cache/chunks"
    # .cache/chunks 디렉토리 생성
    os.makedirs(chunks_folder, exist_ok=True)
    with st.status("Loading video...") as status:
        video_content = video.read()
        safe_video_name = sanitize_filename(video.name)
        video_path = os.path.abspath(os.path.join(".", ".cache", safe_video_name))
        transcript_path = os.path.splitext(video_path)[0] + ".txt"
        try:
            with open(video_path, "wb") as f:
                f.write(video_content)
            logger.info(f"Saved video file: {video_path}")
        except FileNotFoundError as e:
            st.error(f"비디오 파일을 저장하는 중 오류가 발생했습니다: {e}")
            logger.error(f"Error saving video file {video_path}: {e}")
            st.stop()
        status.update(label="Extracting audio...")
        try:
            audio_path = extract_audio_from_video(video_path)
        except Exception as e:
            st.error("오디오 추출에 실패했습니다.")
            logger.error(f"Audio extraction failed: {e}")
            st.stop()
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_path)
    
    # 전사된 텍스트 파일 확인
    if os.path.exists(transcript_path):
        if os.path.getsize(transcript_path) > 0:
            logger.info(f"Transcript file {transcript_path} exists and has content.")
        else:
            logger.error(f"Transcript file {transcript_path} is empty.")
            st.error("전사된 텍스트 파일이 비어 있습니다.")
    else:
        logger.error(f"Transcript file {transcript_path} does not exist.")
        st.error("전사된 텍스트 파일이 생성되지 않았습니다.")

    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tab:
        if os.path.exists(transcript_path):
            if os.path.getsize(transcript_path) > 0:
                try:
                    with open(transcript_path, "r", encoding="utf-8") as file:
                        st.write(file.read())
                except Exception as e:
                    st.error(f"전사된 텍스트를 읽는 중 오류가 발생했습니다: {e}")
                    logger.error(f"Error reading transcript file {transcript_path}: {e}")
            else:
                st.write("전사된 텍스트가 비어 있습니다.")
        else:
            st.write("전사된 텍스트가 없습니다.")

    with summary_tab:
        start = st.button("Generate summary")
        if start:
            try:
                if not os.path.exists(transcript_path) or os.path.getsize(transcript_path) == 0:
                    st.error("전사된 텍스트 파일이 없거나 비어 있습니다.")
                else:
                    loader = TextLoader(transcript_path, encoding='utf-8')
                    docs = loader.load_and_split(text_splitter=splitter)

                    logger.info(f"Loaded {len(docs)} documents for summarization.")
                    for i, doc in enumerate(docs):
                        logger.info(f"Document {i+1}: {doc.page_content[:100]}...")

                    first_summary_prompt = ChatPromptTemplate.from_template(
                        """
                        Write a concise summary of the following:
                        "{text}"
                        CONCISE SUMMARY:                
                    """
                    )

                    first_summary_chain = first_summary_prompt | llm | StrOutputParser()

                    summary = first_summary_chain.invoke(
                        {"text": docs[0].page_content},
                    )

                    refine_prompt = ChatPromptTemplate.from_template(
                        """
                        Your job is to produce a final summary.
                        We have provided an existing summary up to a certain point: {existing_summary}
                        We have the opportunity to refine the existing summary (only if needed) with some more context below.
                        ------------
                        {context}
                        ------------
                        Given the new context, refine the original summary.
                        If the context isn't useful, RETURN the original summary.
                        """
                    )

                    refine_chain = refine_prompt | llm | StrOutputParser()

                    with st.status("Summarizing...") as status:
                        for i, doc in enumerate(docs[1:]):
                            status.update(label=f"Processing document {i+1}/{len(docs)-1} ")
                            summary = refine_chain.invoke(
                                {
                                    "existing_summary": summary,
                                    "context": doc.page_content,
                                }
                            )
                    st.write(summary)
            except Exception as e:
                st.error(f"요약 생성 중 오류가 발생했습니다: {e}")
                logger.error(f"Error generating summary: {e}")

    with qa_tab:
        try:
            retriever = embed_file(transcript_path)
        except Exception as e:
            st.error(f"임베딩 파일 생성 중 오류가 발생했습니다: {e}")
            logger.error(f"Error embedding file {transcript_path}: {e}")
            st.stop()

        # 사용자로부터 질문 입력 받기
        user_question = st.text_input("Ask a question about the transcript:")
        if user_question:
            try:
                # 유사한 문서 검색
                docs = retriever.get_relevant_documents(user_question)
                if docs:
                    # 관련 문서를 기반으로 답변 생성
                    context = "\n\n".join(doc.page_content for doc in docs)
                    qa_prompt = ChatPromptTemplate.from_template(
                        """
                        You are a helpful assistant.

                        Use the following context to answer the question.

                        Context:
                        {context}

                        Question:
                        {question}

                        Answer:
                    """
                    )
                    qa_chain = qa_prompt | llm
                    answer = qa_chain.invoke(
                        {
                            "context": context,
                            "question": user_question,
                        }
                    )
                    st.write(answer)
                else:
                    st.write("No relevant information found.")
            except Exception as e:
                st.error(f"Q&A 생성 중 오류가 발생했습니다: {e}")
                logger.error(f"Error during Q&A: {e}")



#-------ver1.0
# from langchain.storage import LocalFileStore
# import streamlit as st
# import subprocess
# import math
# from pydub import AudioSegment
# import glob
# import openai
# import os
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import StrOutputParser
# from langchain.vectorstores.faiss import FAISS
# from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
# from dotenv import load_dotenv
# import re
# import logging
# from pydub import AudioSegment

# # FFmpeg 실행 파일의 절대 경로 지정
# AudioSegment.converter = r"C:\\Users\\AlvinJang\\Desktop\\report_Maker\\ffmpeg\bin\\ffmpeg.exe"


# # 환경 변수 로드
# load_dotenv()

# # 파일 이름을 안전하게 변환하는 함수
# def sanitize_filename(filename):
#     return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

# # 로깅 설정
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # FFmpeg 설치 여부 확인 함수
# def check_ffmpeg_installed():
#     try:
#         subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         return True
#     except FileNotFoundError:
#         return False

# # FFmpeg가 설치되어 있는지 확인
# if not check_ffmpeg_installed():
#     st.error("FFmpeg가 시스템에 설치되어 있지 않거나, PATH에 포함되지 않았습니다. FFmpeg를 설치하고 다시 시도하세요.")
#     st.stop()

# # OpenAI API 키 로드
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     st.error("OpenAI API 키가 설정되지 않았습니다. `.env` 파일에 `OPENAI_API_KEY`를 추가하세요.")
#     st.stop()

# llm = ChatOpenAI(
#     temperature=0.1,
#     openai_api_key=openai_api_key,  # API 키 로드
# )

# has_transcript = os.path.exists("./.cache/podcast.txt")

# splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=800,
#     chunk_overlap=100,
# )

# @st.cache_data()
# def embed_file(file_path):
#     file_name = os.path.basename(file_path)
#     cache_dir = LocalFileStore(f"./.cache/embeddings/{file_name}")
#     loader = TextLoader(file_path)
#     docs = loader.load_and_split(text_splitter=splitter)
#     embeddings = OpenAIEmbeddings()
#     cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
#     vectorstore = FAISS.from_documents(docs, cached_embeddings)
#     retriever = vectorstore.as_retriever()
#     return retriever

# @st.cache_data()
# def transcribe_chunks(chunk_folder, destination):
#     if has_transcript:
#         return
#     files = glob.glob(f"{chunk_folder}/*.mp3")  # MP3 파일로 유지
#     files.sort()
#     for file in files:
#         with open(file, "rb") as audio_file, open(destination, "a", encoding="utf-8") as text_file:
#             try:
#                 transcript = openai.Audio.transcribe(
#                     "whisper-1",
#                     audio_file,
#                 )
#                 text_file.write(transcript["text"] + "\n")
#                 logger.info(f"Transcribed {file} successfully.")
#             except Exception as e:
#                 logger.error(f"Error transcribing {file}: {e}")
#                 st.error(f"오디오 전사 중 오류가 발생했습니다: {e}")

# @st.cache_data()
# def extract_audio_from_video(video_path):
#     if has_transcript:
#         return
#     audio_path = video_path.rsplit('.', 1)[0] + ".mp3"
#     command = [
#         "ffmpeg",
#         "-y",
#         "-i",
#         video_path,
#         "-vn",
#         audio_path,
#     ]
#     try:
#         subprocess.run(command, check=True)
#         logger.info(f"Audio extracted successfully: {audio_path}")
#     except subprocess.CalledProcessError as e:
#         logger.error(f"FFmpeg 실행 중 오류 발생: {e}")
#         st.error(f"FFmpeg 실행 중 오류가 발생했습니다: {e}")
#         raise e

# @st.cache_data()
# def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
#     if has_transcript:
#         return
#     if not os.path.exists(chunks_folder):
#         os.makedirs(chunks_folder, exist_ok=True)
#     try:
#         track = AudioSegment.from_mp3(audio_path)
#     except Exception as e:
#         st.error(f"오디오 파일을 읽는 중 오류가 발생했습니다: {e}")
#         logger.error(f"Error loading audio file {audio_path}: {e}")
#         return
#     chunk_len = chunk_size * 60 * 1000  # 분을 밀리초로 변환
#     chunks = math.ceil(len(track) / chunk_len)
#     for i in range(chunks):
#         start_time = i * chunk_len
#         end_time = (i + 1) * chunk_len
#         chunk = track[start_time:end_time]
#         chunk_filename = f"chunk_{i}.mp3"
#         try:
#             chunk.export(
#                 os.path.join(chunks_folder, chunk_filename),
#                 format="mp3",
#             )
#             logger.info(f"Exported {chunk_filename}")
#         except Exception as e:
#             st.error(f"오디오 청크를 저장하는 중 오류가 발생했습니다: {e}")
#             logger.error(f"Error exporting chunk {chunk_filename}: {e}")

# # Streamlit 페이지 설정
# st.set_page_config(
#     page_title="MeetingGPT",
#     page_icon="💼",
# )

# st.markdown(
#     """
# # MeetingGPT

# Welcome to MeetingGPT! Upload a video file and I will provide you with a transcript, a summary, and a chatbot to ask any questions about it.

# Get started by uploading a video file in the sidebar.
# """
# )

# with st.sidebar:
#     video = st.file_uploader(
#         "Video",
#         type=["mp4", "avi", "mkv", "mov", "webm"],  # 'webm' 추가
#     )

# if video:
#     chunks_folder = "./.cache/chunks"
#     # .cache/chunks 디렉토리 생성
#     os.makedirs("./.cache/chunks", exist_ok=True)
#     with st.status("Loading video...") as status:
#         video_content = video.read()
#         safe_video_name = sanitize_filename(video.name)
#         video_path = f"./.cache/{safe_video_name}"
#         audio_path = video_path.rsplit('.', 1)[0] + ".mp3"
#         transcript_path = video_path.rsplit('.', 1)[0] + ".txt"
#         try:
#             with open(video_path, "wb") as f:
#                 f.write(video_content)
#         except FileNotFoundError as e:
#             st.error(f"비디오 파일을 저장하는 중 오류가 발생했습니다: {e}")
#             logger.error(f"Error saving video file {video_path}: {e}")
#             st.stop()
#         status.update(label="Extracting audio...")
#         extract_audio_from_video(video_path)
#         status.update(label="Cutting audio segments...")
#         cut_audio_in_chunks(audio_path, 10, chunks_folder)
#         status.update(label="Transcribing audio...")
#         transcribe_chunks(chunks_folder, transcript_path)

#     transcript_tab, summary_tab, qa_tab = st.tabs(
#         [
#             "Transcript",
#             "Summary",
#             "Q&A",
#         ]
#     )

#     with transcript_tab:
#         if os.path.exists(transcript_path):
#             try:
#                 with open(transcript_path, "r", encoding="utf-8") as file:
#                     st.write(file.read())
#             except Exception as e:
#                 st.error(f"전사된 텍스트를 읽는 중 오류가 발생했습니다: {e}")
#                 logger.error(f"Error reading transcript file {transcript_path}: {e}")
#         else:
#             st.write("전사된 텍스트가 없습니다.")

#     with summary_tab:
#         start = st.button("Generate summary")
#         if start:
#             try:
#                 loader = TextLoader(transcript_path)
#                 docs = loader.load_and_split(text_splitter=splitter)

#                 first_summary_prompt = ChatPromptTemplate.from_template(
#                     """
#                     Write a concise summary of the following:
#                     "{text}"
#                     CONCISE SUMMARY:                
#                 """
#                 )

#                 first_summary_chain = first_summary_prompt | llm | StrOutputParser()

#                 summary = first_summary_chain.invoke(
#                     {"text": docs[0].page_content},
#                 )

#                 refine_prompt = ChatPromptTemplate.from_template(
#                     """
#                     Your job is to produce a final summary.
#                     We have provided an existing summary up to a certain point: {existing_summary}
#                     We have the opportunity to refine the existing summary (only if needed) with some more context below.
#                     ------------
#                     {context}
#                     ------------
#                     Given the new context, refine the original summary.
#                     If the context isn't useful, RETURN the original summary.
#                     """
#                 )

#                 refine_chain = refine_prompt | llm | StrOutputParser()

#                 with st.status("Summarizing...") as status:
#                     for i, doc in enumerate(docs[1:]):
#                         status.update(label=f"Processing document {i+1}/{len(docs)-1} ")
#                         summary = refine_chain.invoke(
#                             {
#                                 "existing_summary": summary,
#                                 "context": doc.page_content,
#                             }
#                         )
#                 st.write(summary)
#             except Exception as e:
#                 st.error(f"요약 생성 중 오류가 발생했습니다: {e}")
#                 logger.error(f"Error generating summary: {e}")

#     with qa_tab:
#         try:
#             retriever = embed_file(transcript_path)
#         except Exception as e:
#             st.error(f"임베딩 파일 생성 중 오류가 발생했습니다: {e}")
#             logger.error(f"Error embedding file {transcript_path}: {e}")
#             st.stop()

#         # 사용자로부터 질문 입력 받기
#         user_question = st.text_input("Ask a question about the transcript:")
#         if user_question:
#             try:
#                 # 유사한 문서 검색
#                 docs = retriever.get_relevant_documents(user_question)
#                 if docs:
#                     # 관련 문서를 기반으로 답변 생성
#                     context = "\n\n".join(doc.page_content for doc in docs)
#                     qa_prompt = ChatPromptTemplate.from_template(
#                         """
#                         You are a helpful assistant.

#                         Use the following context to answer the question.

#                         Context:
#                         {context}

#                         Question:
#                         {question}

#                         Answer:
#                     """
#                     )
#                     qa_chain = qa_prompt | llm
#                     answer = qa_chain.invoke(
#                         {
#                             "context": context,
#                             "question": user_question,
#                         }
#                     )
#                     st.write(answer)
#                 else:
#                     st.write("No relevant information found.")
#             except Exception as e:
#                 st.error(f"Q&A 생성 중 오류가 발생했습니다: {e}")
#                 logger.error(f"Error during Q&A: {e}")

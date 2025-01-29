import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import requests
import time

# Function to check URL validity with retries
def check_url(url, retries=3, timeout=10):
    for attempt in range(retries):
        try:
            response = requests.head(url, timeout=timeout)
            if response.status_code == 200:
                return True
            else:
                st.error(f"The provided URL returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                st.warning(f"Attempt {attempt + 1} failed. Retrying... ({str(e)})")
                time.sleep(2)  # Wait before retrying
            else:
                st.error(f"Error connecting to the URL: {e}")
                return False

# Streamlit app configuration
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Get the Groq API Key and URL (YT or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Gemma Model Using Groq API
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Handle button click for summarization
if st.button("Summarize the Content from YT or Website"):
    # Validate inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    elif not check_url(generic_url):  # Check URL validity with retries
        st.error("URL cannot be accessed or is not reachable.")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load the website or YouTube video data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                docs = loader.load()

                # Chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                # Display the summary
                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")

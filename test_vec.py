from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.vectorstores.vectara import Vectara
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
import os

# Load environment variables from .env file
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv('.env')
OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)
VECTARA_CUSTOMER_ID = os.getenv("VECTARA_CUSTOMER_ID")
VECTARA_CORPUS_ID = os.getenv("VECTARA_CORPUS_ID")
VECTARA_API_KEY = os.getenv("VECTARA_API_KEY")
template = """
Based on the provided context, answer every question under each heading:

Context: {context}
question: Carwl carefully, read the full document and completely analyze the provided transcript and provide a highly detailed summary covering the following aspects:
- Introduction: Give an highly detailed overview of the main subjects discussed in the meeting including approaches for productivity.
- Key-Points: Read the document carefully and extract all key points you think are significant, including arguments, problems, solutions,approaches to grab clients and to enhance the productivity level and main discussions.
- Technologies Tools and Sources: Identify and mention any tools, technologies, or sources mentioned during the discussion.
- Conclusion: Summarize the meeting transcript including introduction, any significant key point, importance of this meeting, any marketing/business strategy and it's related outcomes discussed in the meeting
- Future Recommendations: Explain highly any future actions, plans, or recommendations discussed.

{question}

Answer:
"""
# Prompt Template
script_template = PromptTemplate(
    input_variables=['question','context'],
    template=template
)

# Llms
model_name="gpt-3.5-turbo-16k-0613"
llm = ChatOpenAI(temperature=0.8,model_name=model_name, max_tokens=8192)
#script_chain = LLMChain(llm=llm, prompt=script_template,output_key='Answer')

def langchain_fun(file,question):
    loader = TextLoader(file,encoding='utf8')
    documents = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # docs = text_splitter.split_documents(documents)
    vectara = Vectara.from_documents(documents, embedding=OpenAIEmbeddings())
    qa = RetrievalQA.from_llm(llm=llm, prompt=script_template,output_key='Answer', retriever = vectara.as_retriever())
    answer = qa({'query': question,'context': file_uploader})
    formatted_answer = format_answer(answer)
    return formatted_answer
    # results = vectara.similarity_search_with_score(question, n_sentence_context=0)
    # docs, score = results[0]
    # answer = docs.page_content if results else "No results found"
    # print("Answer:", answer)
    # print(f"\nScore: {score}")
def format_answer(answer):
    # Format the answer for better readability
    response = answer['Answer']
    response_lines = response.split('\n')
    formatted_response = "\n".join(line.strip() for line in response_lines)
    return formatted_response

file_uploader = st.file_uploader("Choose a PDF/txt file", accept_multiple_files=True, type=["pdf", "txt"], key="unique_key")

if file_uploader:
    uploaded_file = file_uploader[0]
    file_name = uploaded_file.name
    
    with open(os.path.join(file_name),'wb') as f:
      f.write(uploaded_file.getbuffer())

    question = st.text_area("Enter your question...")
    if st.button("Ask"):
       if len(question) >0:
          answer = langchain_fun(file_name,question)
          st.info("Your question "+ question)
          st.success(answer)





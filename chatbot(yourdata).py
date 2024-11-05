import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.schema import Document

dataset_path = "Your dataset path"

checkpoint = "MBZUAI/LaMini-T5-738M"
persist_directory = "db"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype="auto"
)

def general_llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.1,
        top_p=0.95
    )
    general_llm = HuggingFacePipeline(pipeline=pipe)
    return general_llm

def ingest_data():
    data = pd.read_csv(dataset_path)
    data_texts = data.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()
    documents = [Document(page_content=text) for text in data_texts]
    return documents

def initialize_qa_model(documents):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    retriever = db.as_retriever()
    llm = general_llm_pipeline()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def chat_with_bot(user_input, qa_model, general_llm):
    if any(keyword in user_input.lower() for keyword in ["hello", "hi", "thank", "bye", "how are you"]):
        return general_llm(user_input)
    else:
        query = {'query': user_input}
        result = qa_model(query)
        answer = result['result']
        return answer

documents = ingest_data()
qa_model = initialize_qa_model(documents)
general_llm = general_llm_pipeline()

print("Welcome to the Dataset Chatbot! Ask me about the dataset.")
while True:
       user_input = input("You: ")
       if user_input.lower() in ["exit", "bye"]:
           print("Goodbye!")
           break
       response = chat_with_bot(user_input, qa_model, general_llm)
       print("Bot:", response)
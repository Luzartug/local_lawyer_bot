# IMPORT
import weaviate
import time
import streamlit as st
from pathlib import Path
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from sentence_transformers import SentenceTransformer
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

PATH_CODE_DU_TRAVAIL = Path(__file__).resolve().parent / 'data' / 'code_du_travail.txt'
LLM_MODEL = Path(__file__).resolve().parent / 'model' / 'mistral-7b-instruct-v0.1.Q5_K_S.gguf'
WEAVIATE_URL = "http://localhost:8080"
MODEL_EMBEDDING = "dangvantuan/sentence-camembert-large" 

def main():
    st.set_page_config(
    page_title="Lawyer Bot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
    st.title("Avocat personnel en droit du travail")
    st.info(
        "Posez-moi une question en droit du travail et je vous aiderai à trouver la réponse. 📚",
        icon="ℹ"
    )
    
    llm = load_llm()
    st.write("---------Votre avocat est prêt !----------")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Posez moi une question juridique ?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(get_results(prompt, WEAVIATE_URL, MODEL_EMBEDDING, llm))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    
# Load LLM
def load_llm():
    n_gpu_layers = 10  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
    n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
        model_path=str(LLM_MODEL),
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        f16_kv=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    return llm

# Load data
def load_data(PATH_CODE_DU_TRAVAIL):
    with open(PATH_CODE_DU_TRAVAIL, 'r') as file:
        content = file.read()
        # Assuming paragraphs are separated by two newline characters
        paragraphs = content.split('\n\n')  
        # Trim whitespace and filter out any empty strings
        paragraphs = [p.strip() for p in paragraphs if p.strip() != '']
        return paragraphs

def add_data():
    for law_chunks in laws:
        for sentence in law_chunks:
            # Embed each chunk of the law
            embeddings = model.encode(sentence)
            
            # Concatenate the chunks to form the full law
            #law = ' '.join(law_chunks)
            
            # Create a data object in Weaviate for each law
            data_object = client.data_object.create(
                class_name="Document",
                data_object={
                    "content": sentence
                },
                vector=embeddings.tolist()
            )
    return "Data added to Weaviate"

# Get results
def get_results(query, WEAVIATE_URL, MODEL_EMBEDDING, llm):
    # Instantiate weaviate db
    client = weaviate.Client(WEAVIATE_URL)
    print("---------Weaviate client instantiated----------")
    
    # Load embeddings model
    model = SentenceTransformer(MODEL_EMBEDDING)
    print("---------Embedding model loaded----------")
    
    # Embed the query
    query_embedded = model.encode(query)
    print("---------Query embedded----------")
    
    # Search the context
    response = (
        client.query
        .get('Document', ['content'])
        .with_near_vector({
            "vector": query_embedded
        })
        .with_limit(3)
        .with_additional(["distance"])
        .do()
    )
    context = '\n'.join(document['content'] for document in response['data']['Get']['Document'])
    print("---------Context found----------")
    
    # Template
    template = """Vous êtes un expert en droit français, répondez à cette question en vous basant sur le contexte suivant:
    Contexte: {context}

    Question: {query}
    """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "query"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    #answer = llm_chain.invoke({'context':context,'query':query})
    #return answer['text']
    yield llm_chain.run({'context':context,'query':query})

if __name__ == "__main__":
    main()
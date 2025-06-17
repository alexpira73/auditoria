# Forçando a reconstrução - v1.1
# app.py

import streamlit as st
import json
import os
from google.oauth2.service_account import Credentials
from langchain_google_community.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Agente de Auditoria Médica", page_icon="🤖", layout="wide")
st.title("🤖 Agente de IA para Auditoria e Regulação Médica")
st.write("Desenvolvido por Nefrologia para análise de solicitações com base em diretrizes.")

# --- FUNÇÕES EM CACHE PARA EFICIÊNCIA ---

# Função para carregar credenciais da Conta de Serviço (executa só uma vez)
@st.cache_resource
def load_gdrive_credentials():
    # Pega as credenciais JSON a partir do secrets do Streamlit
    creds_json = json.loads(st.secrets["gcp_service_account_json"])
    scopes = ['https://www.googleapis.com/auth/drive']
    return Credentials.from_service_account_info(creds_json, scopes=scopes)

# Função para carregar e processar os documentos (executa só uma vez)
@st.cache_resource
def load_and_process_documents():
    try:
        credentials = load_gdrive_credentials()
        folder_id = st.secrets["google_drive_folder_id"]
        
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            credentials=credentials,
            recursive=False
        )
        docs = loader.load()

        if not docs:
            st.error("Nenhum documento foi encontrado na pasta do Google Drive. Verifique o ID da pasta e as permissões de compartilhamento.")
            return None, None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        return vectorstore
    except Exception as e:
        st.error(f"Erro ao carregar ou processar documentos: {e}")
        return None

# Função para criar a cadeia de conversação (executa só uma vez)
@st.cache_resource
def get_conversational_chain(_vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1)
    
    prompt = ChatPromptTemplate.from_template("""
    Você é um assistente especialista em auditoria e regulação médica. Sua única fonte de conhecimento são os documentos fornecidos.
    Responda a pergunta do usuário de forma objetiva e técnica, baseando-se SOMENTE no contexto abaixo.
    Se a informação não estiver no contexto, responda: "Não encontrei informações sobre isso nas diretrizes fornecidas."
    Ao final da sua resposta, cite o nome do documento fonte de onde você tirou a informação, se disponível.

    Contexto Fornecido:
    {context}

    Pergunta do Usuário:
    {input}

    Resposta Fundamentada:
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(_vectorstore.as_retriever(), document_chain)

# --- INICIALIZAÇÃO DA APLICAÇÃO ---
try:
    # Carrega o Gemini API Key
    os.environ["GOOGLE_API_KEY"] = st.secrets["google_api_key"]

    vectorstore = load_and_process_documents()

    if vectorstore:
        st.success("Base de conhecimento carregada e pronta!")
        chain = get_conversational_chain(vectorstore)

        # --- INTERFACE DO USUÁRIO ---
        st.subheader("Análise de Solicitação")
        
        col1, col2 = st.columns(2)
        with col1:
            solicitacao_input = st.text_input("Procedimento/Exame Solicitado:")
        with col2:
            idade_input = st.number_input("Idade do Paciente:", min_value=0, max_value=120)

        contexto_input = st.text_area("Contexto Clínico e Justificativa Médica:", height=150)

        if st.button("Analisar Caso", type="primary"):
            if not solicitacao_input or not contexto_input:
                st.warning("Por favor, preencha todos os campos para a análise.")
            else:
                pergunta_completa = f"""
                Analise a seguinte solicitação de auditoria:
                - Solicitação do Exame/Procedimento: {solicitacao_input}
                - Idade do Paciente: {idade_input}
                - Contexto Clínico do Paciente: {contexto_input}
                
                Verifique se esta solicitação está em conformidade com as diretrizes de utilização.
                """
                with st.spinner("Analisando com base nas diretrizes..."):
                    response = chain.invoke({"input": pergunta_completa})
                    st.divider()
                    st.subheader("Resultado da Análise")
                    st.markdown(response["answer"])

except KeyError as e:
    st.error(f"Erro de configuração: A chave secreta '{e.name}' não foi encontrada. Por favor, configure os secrets no Streamlit Community Cloud.")
except Exception as e:
    st.error(f"Ocorreu um erro inesperado na aplicação: {e}")

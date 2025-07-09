import os
import json
import gradio as gr
from google.oauth2.service_account import Credentials
from langchain_community.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- CARREGAMENTO E PROCESSAMENTO (LÓGICA DO AGENTE) ---
# Esta parte é executada apenas uma vez quando o aplicativo inicia.

try:
    # Carrega as credenciais a partir dos secrets do Hugging Face
    creds_json_str = os.environ["gcp_service_account_json"]
    creds_json = json.loads(creds_json_str)
    scopes = ['https://www.googleapis.com/auth/drive']
    credentials = Credentials.from_service_account_info(creds_json, scopes=scopes)

    folder_id = os.environ["google_drive_folder_id"]

    # Carrega os documentos do Google Drive
    loader = GoogleDriveLoader(
        folder_id=folder_id,
        credentials=credentials,
        recursive=False
    )
    docs = loader.load()

    # Divide e indexa os documentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)

    # Cria a cadeia de conversação
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.1,
        google_api_key=os.environ["google_api_key"]
    )

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
    chain = create_retrieval_chain(vectorstore.as_retriever(), document_chain)

    # Flag para indicar que o setup foi bem-sucedido
    setup_success = True
except Exception as e:
    setup_success = False
    # Guardamos o erro para mostrar na interface
    setup_error = e

# --- FUNÇÃO PRINCIPAL QUE SERÁ USADA PELA INTERFACE ---
def analisar_solicitacao(solicitacao, contexto_clinico):
    if not setup_success:
        return f"ERRO NA INICIALIZAÇÃO DO AGENTE: {setup_error}"

    if not solicitacao or not contexto_clinico:
        return "Por favor, preencha ambos os campos: Procedimento Solicitado e Contexto Clínico."

    pergunta_completa = f"""
    Analise a seguinte solicitação de auditoria:
    - Solicitação do Exame/Procedimento: {solicitacao}
    - Contexto Clínico do Paciente: {contexto_clinico}
    Verifique se esta solicitação está em conformidade com as diretrizes de utilização.
    """

    response = chain.invoke({"input": pergunta_completa})
    return response["answer"]

# --- CRIAÇÃO DA INTERFACE COM O GRADIO ---
iface = gr.Interface(
    fn=analisar_solicitacao,
    inputs=[
        gr.Textbox(lines=2, placeholder="Ex: Ressonância Magnética do Ombro", label="Procedimento Solicitado"),
        gr.Textbox(lines=5, placeholder="Ex: Paciente de 45 anos, com dor há 3 meses, sem resposta a tratamento conservador...", label="Contexto Clínico e Justificativa")
    ],
    outputs=gr.Markdown(label="Análise do Agente"),
    title="🤖 Agente de IA para Auditoria Médica",
    description="Desenvolvido por Nefrologia. Insira os dados da solicitação para que o agente analise com base nas diretrizes carregadas."
)

if __name__ == "__main__":
    iface.launch()

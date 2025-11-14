##  RAG com DeepSeek — `dados_deepseek`

Este diretório contém o script e os arquivos necessários para executar os testes de **RAG (Retrieval-Augmented Generation)** utilizando o modelo **Gemini** via **LangChain**.

O script principal é: rag_tester_gemini.py

-Lê os CSVs (empresa, produtos, servicos, questions_padronizadas);
-Converte cada linha em Document com metadados;
-Gera embeddings com MiniLM (sentence-transformers/all-MiniLM-L6-v2);
-Cria ou reutiliza o índice vetorial ChromaDB (chroma_index/);
-Recupera os documentos mais relevantes (Top-K) para cada pergunta;
-Envia o contexto + pergunta para o DeepSeek;
-Salva as respostas, latência e tokens em resultados_rag.csv.

##  Dependências

Versão recomendada: Python 3.10+

Instalar as bibliotecas:

pip install langchain langchain-google-genai google-generativeai \
                langchain-community chromadb faiss-cpu tiktoken pandas sentence-transformers



## 🔑 Variáveis de ambiente

O script lê as seguintes variáveis:
GOOGLE_API_KEY – obrigatória (chave da API Gemini)
GEMINI_CHAT_MODEL – opcional (padrão: gemini-2.0-flash ou gemini-2.5-flash)
RAG_TOPK – opcional (padrão: 26 documentos recuperados)

## ▶️ Como executar

Dentro do terminal execute

python .\rag_tester_gemini.py


Se tudo estiver correto, o terminal exibirá algo como:

Carregando base...
Documentos base: XXX
Índice vetorial (Chroma)...
OK 1: 0.XXXs
OK 2: 0.XXXs
...
Resultados salvos em resultados_rag.csv
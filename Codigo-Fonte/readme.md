## 📁 Código-Fonte do Projeto — TCC 2025
Análise Comparativa de APIs de IA para Integração com Sistemas Empresariais

Este diretório reúne todo o material técnico utilizado nos experimentos do TCC, incluindo:

bases de dados usadas para testes,

scripts Python para execução dos cenários RAG,

resultados extraídos (via Python e Postman),

estruturas organizacionais do projeto,

e documentação específica de cada API testada.

O objetivo deste diretório é permitir a reprodutibilidade completa dos testes realizados com OpenAI GPT, Google Gemini e DeepSeek.

Codigo-Fonte/
│
├── TCC.PROJ/                        # Arquivos de organização do projeto
│
├── dados_gpt/                       # Testes RAG com GPT (OpenAI)
│   └── readme.md                    # Instruções específicas
│
├── dados_gemini/                    # Testes RAG com Gemini (Google)
│   └── readme.md                    # Instruções específicas
│
├── dados_deepseek/                  # Testes RAG com DeepSeek
│   └── readme.md                    # Instruções específicas
│
├── resultados Python gpt/           # Resultados dos testes GPT 3.5 Turbor e 4.o Turbo via Python
├── resultados Python gemini/        # Resultados dos testes Gemini 2.0 Flash e 2.5 Flash via Python
├── resultados Python deepseek/      # Resultados dos testes DeepSeek Chat via Python
│
├── CONTEXTO_LOJA.txt                # Contexto utilizado nos testes Python e Postman
├── PERGUNTAS TESTES.txt             # Perguntas padronizadas para testes comparativos
│
├── RESULTADOS TESTES - POSTMAN.zip  # Retornos completos dos testes via Postman
└── Resultados Tabelados - Python-Postman.xlsx         # Consolidação de latência, tokens e desempenho


## Observação Final

As instruções detalhadas (dependências, variáveis de ambiente e execução) estão dentro de cada pasta de API, para facilitar a manutenção e deixar o repositório mais organizado.
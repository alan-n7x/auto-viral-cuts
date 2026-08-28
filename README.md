# 🚀 Auto Viral Cuts

> Aplicação profissional e modular em Python para geração automática de cortes virais (TikTok, Reels, Shorts) a partir de vídeos longos usando a **API do Google Gemini** e **FFmpeg**.

---

## 🏗️ Arquitetura do Projeto (`src` layout)

O projeto foi construído seguindo os mais altos padrões de engenharia de software, separando a lógica de negócio central (IA e processamento de vídeo) da interface do usuário (Gradio) e da API REST (FastAPI).

```text
auto-viral-cuts/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── gemini_analyzer.py   # Upload, polling de estado da File API e análise estruturada com Pydantic
│   │   ├── video_processor.py   # Execução FFmpeg (crop 9:16, cortes precisos, áudio AAC)
│   │   └── schemas.py           # Modelos Pydantic (ClipMetadata, ProcessingOptions, ProcessingResult)
│   ├── ui/
│   │   ├── __init__.py
│   │   └── gradio_app.py        # Interface visual com Blocks e Drag & Drop
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py            # Endpoints REST iniciais no FastAPI
│   └── main.py                  # Entrypoint principal (FastAPI + Gradio montado)
├── setup_issues.sh              # Script Bash para criar automaticamente as GitHub Issues via GitHub CLI
├── output_cuts/                 # Diretório local para salvar cortes (ignorado no git)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ✨ Funcionalidades Principais

1. **Análise Inteligente com Google Gemini 2.5 Flash:**
   - Upload de vídeos longos via Google File API com polling automático de estado.
   - Extração estruturada de momentos virais com suporte a Pydantic (`response_schema`).
   - Retorno de ganchos (hooks), pontuação de virailidade (0-100), legendas prontas e hashtags estratégicas.

2. **Processamento de Vídeo Preciso com FFmpeg:**
   - Corte de alta precisão com seek otimizado.
   - Conversão para formato vertical 9:16 através de múltiplos modos de enquadramento (`center_crop`, `blurred_background`, `fit_black_bars`, `no_crop`).
   - Normalização de áudio AAC estéreo e codecs otimizados para redes sociais.

3. **Interface Gráfica Drag & Drop (Gradio):**
   - Interface web moderna e intuitiva para uso local imediato.
   - Pré-visualização do primeiro corte gerado e relatório detalhado.

4. **Arquitetura Web / SaaS Pronta para Produção (FastAPI):**
   - Endpoints REST para integração com frontends modernos e microsserviços.

---

## 🛠️ Instalação e Execução Local

### Pré-requisitos
- Python 3.10+
- FFmpeg instalado no sistema (`sudo apt install ffmpeg` no Ubuntu/Debian ou `brew install ffmpeg` no macOS).
- Chave de API do Google Gemini ([Obtenha gratuitamente no AI Studio](https://aistudio.google.com/)).

### 1. Clonar e Configurar o Ambiente
```bash
git clone https://github.com/alan-n7x/auto-viral-cuts.git
cd auto-viral-cuts

# Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Configurar Variáveis de Ambiente
Copie o arquivo de exemplo e insira sua chave da API do Gemini:
```bash
cp .env.example .env
```
Edite o arquivo `.env` e preencha `GEMINI_API_KEY=sua_chave_aqui`.

### 3. Executar a Aplicação
Inicie a aplicação unificada (FastAPI + Gradio):
```bash
python3 src/main.py
```
- Acesse a interface web em: **`http://localhost:8000/ui`**
- Acesse a documentação da API REST em: **`http://localhost:8000/docs`**

---

## 📋 Automação de Issues no GitHub

O projeto inclui um script Bash (`setup_issues.sh`) para popular automaticamente o repositório com issues de planejamento e roadmap utilizando o GitHub CLI (`gh`).

Para executar:
```bash
chmod +x setup_issues.sh
./setup_issues.sh
```

---

## 📜 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.

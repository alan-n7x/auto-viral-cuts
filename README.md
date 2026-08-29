# Auto Viral Cuts

> Aplicação profissional e modular em Python para geração automática de cortes virais (TikTok, Reels, Shorts) a partir de vídeos longos usando a **API do Google Gemini** e **FFmpeg**.

---

## Arquitetura do Projeto (`src` layout)


O projeto foi construído seguindo os mais altos padrões de engenharia de software, separando a lógica de negócio central (IA e processamento de vídeo) da interface do usuário (Gradio) e da API REST (FastAPI).

```text
auto-viral-cuts/
├── src/
│   ├── __init__.py
│   ├── domain/                  # Camada de Domínio (Clean Architecture)
│   │   └── ports/
│   │       └── video_processor_port.py  # Porta abstrata (Contrato DIP)
│   ├── application/             # Camada de Aplicação (Use Cases & Jobs)
│   │   ├── task_manager.py      # Gerenciador thread-safe de tarefas em background
│   │   └── use_cases/
│   │       └── process_video_use_case.py # Caso de uso de processamento desacoplado
│   ├── infrastructure/          # Camada de Infraestrutura (Adapters)
│   │   └── adapters/
│   │       └── local_processor_adapter.py # Adaptador local (FFmpeg, Whisper, Gemini)
│   ├── core/
│   │   ├── gemini_analyzer.py   # Upload otimizado, proxy leve e análise Gemini
│   │   ├── video_processor.py   # Motor FFmpeg (1080x1920, GPU VAAPI scale_vaapi)
│   │   ├── transcriber.py       # Transcrição com timestamps a nível de palavra
│   │   ├── subtitle_generator.py # Geração de legendas ASS estilizadas
│   │   └── schemas.py           # Modelos Pydantic (ClipMetadata, TaskState, etc.)
│   ├── ui/
│   │   ├── client_renderer/     # Studio WebCodecs 9:16 no navegador do cliente
│   │   └── gradio_app.py        # Interface visual com Blocks e Drag & Drop
│   ├── api/
│   │   ├── dependencies.py      # Injeção manual de dependência no FastAPI
│   │   └── routes.py            # Endpoints REST (streaming aiofiles 64KB, 202 Accepted)
│   └── main.py                  # Entrypoint principal (FastAPI + Gradio + Studio)
├── tests/                       # 30 testes unitários e de integração com pytest
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Funcionalidades Principais

1. **Análise Inteligente com Google Gemini 3.6 Flash:**
   - Upload de vídeos longos via Google File API com polling automático de estado.
   - Extração estruturada de momentos virais pela Interactions API com JSON Schema do Pydantic.
   - Retorno de ganchos (hooks), pontuação de virailidade (0-100), legendas prontas e hashtags estratégicas.

2. **Processamento de Vídeo Preciso com FFmpeg:**
   - Corte de alta precisão com seek otimizado.
   - Conversão para formato vertical 9:16 através de múltiplos modos de enquadramento (`center_crop`, `blurred_background`, `fit_black_bars`, `no_crop`).
   - Normalização de áudio AAC estéreo e codecs otimizados para redes sociais.

3. **Studio de Renderização no Cliente (WebCodecs + Canvas 2D):**
   - Arquitetura híbrida de ultra-baixo custo: upload apenas da faixa de áudio leve para IA (`/api/v1/generate-manifest`).
   - Corte 9:16, renderização de legendas estilo Hormozi e exportação MP4 realizadas 100% no navegador do usuário via **WebCodecs** (`VideoEncoder` com aceleração por hardware) e **`mp4-muxer`**.
   - Disponível diretamente em: **`http://localhost:8000/client`**.

4. **Interface Gráfica Servidor (Gradio):**
   - Interface web clássica com processamento local via FFmpeg/VAAPI em: **`http://localhost:8000/ui`**.
   - Pré-visualização do primeiro corte gerado e relatório detalhado.

5. **Arquitetura Web / SaaS Pronta para Produção (FastAPI):**
   - Endpoints REST documentados interativamente em: **`http://localhost:8000/docs`**.

---

## Instalação e Execução Local

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
Inicie a aplicação unificada (FastAPI + Client Studio + Gradio):
```bash
python3 src/main.py
```
- Acesse o **Studio WebCodecs (Cliente)** em: **`http://localhost:8000/client`**
- Acesse a **Interface Servidor (Gradio)** em: **`http://localhost:8000/ui`**
- Acesse a **Documentação da API REST** em: **`http://localhost:8000/docs`**


---

## Automação de Issues no GitHub

O projeto inclui um script Bash (`setup_issues.sh`) para popular automaticamente o repositório com issues de planejamento e roadmap utilizando o GitHub CLI (`gh`).

Para executar:
```bash
chmod +x setup_issues.sh
./setup_issues.sh
```

---

## Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.


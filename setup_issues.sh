#!/usr/bin/env bash
# ==============================================================================
# Script para automação da criação de GitHub Issues usando a GitHub CLI (gh)
# ==============================================================================

set -e

echo "🚀 Verificando autenticação do GitHub CLI (gh)..."
if ! gh auth status &>/dev/null; then
    echo "❌ Erro: O GitHub CLI (gh) não está autenticado. Execute 'gh auth login' primeiro."
    exit 1
fi

echo "📋 Criando GitHub Issues para o projeto Auto Viral Cuts..."

gh issue create \
    --title "feat: Implementar suporte a legendas automáticas embutidas (SRT/ASS)" \
    --label "enhancement" \
    --body "Adicionar geração automática de legendas estilizadas utilizando Whisper ou Gemini transcrições e burn-in via FFmpeg (ass/srt)."

gh issue create \
    --title "feat: Adicionar processamento assíncrono com Celery e Redis para SaaS" \
    --label "enhancement" \
    --body "Migrar o processamento de vídeos longos de síncrono para workers em background com Celery + Redis para suportar alto volume de requisições na API."

gh issue create \
    --title "refactor: Otimizar velocidade de compressão FFmpeg com aceleração por GPU (NVENC/VideoToolbox)" \
    --label "enhancement" \
    --body "Detectar automaticamente hardware gráfico disponível (NVIDIA NVENC, Apple Silicon VideoToolbox) para acelerar a renderização dos cortes."

gh issue create \
    --title "test: Adicionar testes unitários e de integração com pytest" \
    --label "enhancement" \
    --body "Criar suite de testes cobrindo os parsers de timestamp, chamadas mockadas do Gemini e renderização FFmpeg."

echo "✅ Todas as GitHub Issues foram criadas com sucesso!"

@echo off
REM Wrapper para Windows - executa o orquestrador da Etapa 1.
REM Repassa quaisquer argumentos para run_etapa1.py via uv.

setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERRO] uv nao encontrado no PATH. Instale com: pip install uv
    echo        Documentacao: https://docs.astral.sh/uv/
    exit /b 1
)

uv run python run_etapa1.py %*
exit /b %errorlevel%

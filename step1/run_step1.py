"""Orquestrador da Etapa 1 do Tech Challenge.

Executa, em sequência:
  1. Verificação do dataset
  2. Sincronização do ambiente Python via uv (uv sync)
  3. Subida do servidor MLflow via docker compose
  4. Health check do MLflow em http://localhost:5000
  5. Execução dos notebooks (EDA + Baselines) com nbconvert dentro do venv do uv

Uso:
    python run_step1.py                     # roda tudo
    python run_step1.py --skip-deps         # pula uv sync
    python run_step1.py --skip-docker       # assume MLflow já rodando
    python run_step1.py --skip-notebooks    # apenas sobe a infra
    python run_step1.py --shutdown          # derruba os containers ao final
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Garante UTF-8 no console (especialmente no Windows com cp1252).
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (AttributeError, OSError):
                pass

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATASET_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
NOTEBOOKS = [
    ROOT / "notebooks" / "01_eda_analysis.ipynb",
    ROOT / "notebooks" / "02_baselines.ipynb",
]
MLFLOW_URL = "http://localhost:5000"
# 600s para acomodar a primeira execução (pip install dentro do container ~ 2-3min).
MLFLOW_HEALTH_TIMEOUT = 600
NB_EXEC_TIMEOUT = 900

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("step1")


class StepFailed(RuntimeError):
    """Levantada quando uma etapa do pipeline falha."""


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    """Executa um comando, propagando stdout/stderr e abortando em caso de erro."""
    log.info("$ %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise StepFailed(f"Comando falhou ({result.returncode}): {' '.join(cmd)}")


def ensure_uv() -> None:
    if shutil.which("uv") is None:
        raise StepFailed(
            "uv não está instalado ou não está no PATH.\n"
            "Instale com: pip install uv  (ou veja https://docs.astral.sh/uv/)"
        )


def detect_compose() -> list[str]:
    """Retorna o prefixo de comando do docker compose disponível."""
    if shutil.which("docker") is None:
        raise StepFailed(
            "Docker não está instalado ou não está no PATH. "
            "Instale Docker Desktop: https://www.docker.com/products/docker-desktop"
        )
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            check=True,
            capture_output=True,
        )
        return ["docker", "compose"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    raise StepFailed(
        "Nem 'docker compose' (v2) nem 'docker-compose' (v1) estão disponíveis."
    )


def ensure_dataset() -> None:
    if not DATASET_PATH.exists():
        raise StepFailed(
            f"Dataset não encontrado em {DATASET_PATH}\n"
            "Coloque o arquivo Telco-Customer-Churn.csv dentro da pasta data/."
        )
    size_kb = DATASET_PATH.stat().st_size / 1024
    log.info("Dataset OK: %s (%.1f KB)", DATASET_PATH.name, size_kb)


def ensure_pyproject() -> None:
    if not PYPROJECT.exists():
        raise StepFailed(
            f"pyproject.toml não encontrado em {PYPROJECT}.\n"
            "Esperado na raiz do projeto (uma pasta acima de step1/)."
        )


def uv_sync() -> None:
    log.info("Sincronizando ambiente com uv (uv sync)")
    run_cmd(["uv", "sync"], cwd=PROJECT_ROOT)


def start_mlflow(compose: list[str]) -> None:
    log.info("Subindo MLflow via docker compose")
    run_cmd([*compose, "up", "-d", "mlflow"], cwd=ROOT)


def wait_mlflow(timeout: int = MLFLOW_HEALTH_TIMEOUT) -> None:
    """Aguarda o MLflow responder em :5000 com retries tolerantes.

    Durante a inicialização (pip install + boot), o Docker pode aceitar a conexão
    na porta exposta e abortar logo em seguida (ConnectionAbortedError no Windows,
    ConnectionResetError em Linux). Capturamos OSError, que é a classe-mãe de
    todos esses erros de socket — incluindo ConnectionRefusedError, TimeoutError,
    ConnectionAbortedError e ConnectionResetError.
    """
    log.info("Aguardando MLflow responder em %s (timeout: %ds)", MLFLOW_URL, timeout)
    deadline = time.time() + timeout
    last_error: str | None = None
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(f"{MLFLOW_URL}/health", timeout=3) as resp:
                if resp.status == 200:
                    log.info("MLflow está pronto.")
                    return
                last_error = f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # MLflow respondeu, mas /health não está habilitado em algumas versões.
                log.info("MLflow respondendo em %s (sem endpoint /health).", MLFLOW_URL)
                return
            last_error = f"HTTPError {exc.code}"
        except urllib.error.URLError as exc:
            last_error = f"URLError ({exc.reason})"
        except OSError as exc:
            last_error = f"{type(exc).__name__} ({exc})"
        # Logs intermediários a cada ~30s para o usuário saber que ainda estamos esperando.
        if attempt % 10 == 0:
            log.info("... ainda aguardando MLflow (último erro: %s)", last_error)
        time.sleep(3)
    raise StepFailed(
        f"MLflow não ficou disponível em {timeout}s. Último erro: {last_error}"
    )


def execute_notebook(path: Path) -> None:
    """Executa um notebook usando o Python do venv do uv."""
    log.info("Executando notebook: %s", path.relative_to(ROOT))
    run_cmd(
        [
            "uv",
            "run",
            "--",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.timeout={NB_EXEC_TIMEOUT}",
            str(path),
        ],
        cwd=PROJECT_ROOT,
    )


def shutdown_stack(compose: list[str]) -> None:
    log.info("Derrubando containers (docker compose down)")
    run_cmd([*compose, "down"], cwd=ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-deps", action="store_true", help="Não executa uv sync")
    parser.add_argument("--skip-docker", action="store_true", help="Assume MLflow já rodando")
    parser.add_argument("--skip-notebooks", action="store_true", help="Não executa os notebooks")
    parser.add_argument("--shutdown", action="store_true", help="Encerra os containers ao final")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log.info("=== Tech Challenge — Etapa 1 ===")
    compose: list[str] | None = None

    try:
        ensure_dataset()
        ensure_pyproject()
        ensure_uv()

        if not args.skip_deps:
            uv_sync()
        else:
            log.info("Pulando sincronização de dependências (--skip-deps).")

        if not args.skip_docker:
            compose = detect_compose()
            start_mlflow(compose)
            wait_mlflow()
        else:
            log.info("Pulando Docker (--skip-docker). Verificando se MLflow está acessível...")
            wait_mlflow(timeout=15)

        if not args.skip_notebooks:
            for nb in NOTEBOOKS:
                execute_notebook(nb)
        else:
            log.info("Pulando execução dos notebooks (--skip-notebooks).")

    except StepFailed as exc:
        log.error("Falha na execução: %s", exc)
        return 1
    except KeyboardInterrupt:
        log.warning("Interrompido pelo usuário.")
        return 130

    log.info("=" * 60)
    log.info("✓ Etapa 1 concluída com sucesso.")
    log.info("MLflow UI:    %s", MLFLOW_URL)
    log.info("Notebooks:    step1/notebooks/01_eda_analysis.ipynb")
    log.info("              step1/notebooks/02_baselines.ipynb")
    log.info("Experimento:  TelcoChurn_Step1_Baselines")
    log.info("=" * 60)

    if args.shutdown and compose is not None:
        shutdown_stack(compose)

    return 0


if __name__ == "__main__":
    sys.exit(main())

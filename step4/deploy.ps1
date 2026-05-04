<#
.SYNOPSIS
    Deploy step4 services to Railway.

.DESCRIPTION
    Deploys mlflow, trainer, and api from the custom Dockerfile.
    Prometheus and Grafana use official Docker Hub images — see notes at the end.

    Prerequisites:
      - Railway CLI:  npm install -g @railway/cli
      - Logged in:    railway login
      - Project linked or use -Init to create a new one

.PARAMETER Init
    Create a new Railway project instead of linking to an existing one.

.PARAMETER ProjectName
    Name for the new project when using -Init. Default: fiap-step4.

.PARAMETER SkipData
    Skip copying the dataset into the build context (use if already present).

.EXAMPLE
    # First-time setup
    .\deploy.ps1 -Init

    # Re-deploy after code changes
    .\deploy.ps1
#>

param(
    [switch]$Init,
    [string]$ProjectName = "fiap-step4",
    [switch]$SkipData
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Step4Dir  = $PSScriptRoot
$DataDest  = Join-Path $Step4Dir "data"
$DataSrc   = Join-Path $Step4Dir "..\data\Telco-Customer-Churn.csv"

# ── Helpers ────────────────────────────────────────────────────────────────────

function Write-Step([string]$msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

function Write-Ok([string]$msg) {
    Write-Host "    ✓ $msg" -ForegroundColor Green
}

function Write-Warn([string]$msg) {
    Write-Host "    ⚠ $msg" -ForegroundColor Yellow
}

function Invoke-Railway([string[]]$args) {
    $result = & railway @args 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "railway $($args -join ' ') failed:`n$result"
    }
    return $result
}

# ── 1. Prerequisites ───────────────────────────────────────────────────────────

Write-Step "Checking prerequisites"

if (-not (Get-Command railway -ErrorAction SilentlyContinue)) {
    Write-Error "Railway CLI not found. Install it: npm install -g @railway/cli"
    exit 1
}
Write-Ok "Railway CLI found: $(railway --version 2>&1)"

# ── 2. Dataset ─────────────────────────────────────────────────────────────────

Write-Step "Preparing dataset for Docker build context"

if (-not $SkipData) {
    if (Test-Path $DataSrc) {
        New-Item -ItemType Directory -Force -Path $DataDest | Out-Null
        Copy-Item -Path $DataSrc -Destination $DataDest -Force
        Write-Ok "Copied Telco-Customer-Churn.csv → step4/data/"
    } elseif (Test-Path (Join-Path $DataDest "Telco-Customer-Churn.csv")) {
        Write-Ok "Dataset already present in step4/data/"
    } else {
        Write-Error "Dataset not found at '$DataSrc'. Place Telco-Customer-Churn.csv in step4/data/ manually."
        exit 1
    }
} else {
    Write-Ok "Skipped (--SkipData)"
}

# ── 3. Railway project ─────────────────────────────────────────────────────────

Write-Step "Setting up Railway project"

Set-Location $Step4Dir

if ($Init) {
    Invoke-Railway @("init", "--name", $ProjectName) | Out-Null
    Write-Ok "Project '$ProjectName' created"
} else {
    Invoke-Railway @("link") | Out-Null
    Write-Ok "Linked to existing project"
}

# ── 4. Service definitions ─────────────────────────────────────────────────────

$services = @(
    @{
        Name = "mlflow"
        Vars = @{
            SERVICE = "mlflow"
            PORT    = "5000"
        }
    },
    @{
        Name = "trainer"
        Vars = @{
            SERVICE     = "trainer"
            MODELS_PATH = "/models"
            # ML_FLOW_API is set after mlflow is deployed (see step 6)
        }
    },
    @{
        Name = "api"
        Vars = @{
            SERVICE     = "api"
            PORT        = "8000"
            MODELS_PATH = "/models"
            ENVIRONMENT = "production"
        }
    }
)

# ── 5. Configure env vars ──────────────────────────────────────────────────────

Write-Step "Configuring environment variables"

foreach ($svc in $services) {
    Write-Host "    $($svc.Name)" -ForegroundColor White
    railway service $svc.Name

    foreach ($kv in $svc.Vars.GetEnumerator()) {
        Invoke-Railway @("variables", "set", "$($kv.Key)=$($kv.Value)") | Out-Null
        Write-Host "      $($kv.Key) = $($kv.Value)"
    }
}

# ── 6. Inter-service references ────────────────────────────────────────────────
# Railway internal networking: {service}.railway.internal:{port}

Write-Step "Setting inter-service references"

railway service trainer
Invoke-Railway @("variables", "set", "ML_FLOW_API=http://mlflow.railway.internal:5000") | Out-Null
Write-Ok "trainer → ML_FLOW_API"

railway service api
Invoke-Railway @("variables", "set", "ML_FLOW_API=http://mlflow.railway.internal:5000") | Out-Null
Write-Ok "api → ML_FLOW_API"

# ── 7. Deploy in dependency order ──────────────────────────────────────────────

Write-Step "Deploying services"

$deployOrder = @("mlflow", "trainer", "api")
foreach ($name in $deployOrder) {
    Write-Host "    Deploying $name..." -ForegroundColor White
    railway service $name
    Invoke-Railway @("up", "--detach") | Out-Null
    Write-Ok "$name deployment triggered"
}

# ── 8. Done ────────────────────────────────────────────────────────────────────

Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║  Deployment triggered. Monitor with:                         ║
║    railway logs --service mlflow                             ║
║    railway logs --service trainer                            ║
║    railway logs --service api                                ║
╚══════════════════════════════════════════════════════════════╝

MANUAL STEPS — Prometheus and Grafana:
  Railway does not auto-deploy official Docker Hub images from this script.
  Add them via the Railway dashboard → New Service → Docker Image:

    prometheus:
      Image:   prom/prometheus:v2.53.0
      Env:     (none required)
      Volume:  mount monitoring/prometheus.yml → /etc/prometheus/prometheus.yml
      NOTE:    Update monitoring/prometheus.yml target from 'api:8000'
               to 'api.railway.internal:8000'

    grafana:
      Image:   grafana/grafana:11.1.0
      Env:     GF_SECURITY_ADMIN_USER=admin
               GF_SECURITY_ADMIN_PASSWORD=<your-password>
               GF_USERS_ALLOW_SIGN_UP=false

SHARED VOLUME WARNING:
  trainer writes the model to MODELS_PATH=/models
  api reads the model from MODELS_PATH=/models
  Railway volumes are per-service (not shared).
  Run trainer FIRST and let it complete before api starts.
  Both services need a Railway persistent volume mounted at /models.
  Configure in dashboard: Service → Settings → Volumes → Add Volume → /models

"@ -ForegroundColor Yellow

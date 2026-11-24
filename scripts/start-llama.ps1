[CmdletBinding()]
param(
    [string]$LlamaRoot = "$env:LOCALAPPDATA\Programs\llama.cpp",
    [string]$ModelPath,
    [string]$Server,
    [string]$Alias = "gemma-4-E4B-local",
    [ValidateRange(1,65535)][int]$Port = 9940,
    [ValidateRange(1024,131072)][int]$Context = 8192,
    [ValidateRange(1,16)][int]$Parallel = 4,
    [string]$GpuLayers = "all"
)
$ErrorActionPreference = "Stop"
if (-not $ModelPath) {
    $ModelPath = Join-Path $LlamaRoot "models\gemma-4-E4B-it-UD-Q8_K_XL.gguf"
}
if (-not $Server) {
    $Server = Join-Path $LlamaRoot "releases\b12242-cuda13.5\llama-server.exe"
    if (-not (Test-Path -LiteralPath $Server -PathType Leaf)) {
        $Server = Join-Path $LlamaRoot "llama-server.exe"
    }
}
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) { throw "GGUF missing at $ModelPath" }
if (-not (Test-Path -LiteralPath $Server -PathType Leaf)) { throw "llama-server missing at $Server" }
if ($Context -lt 2048 * $Parallel) { throw "Allocate at least 2048 context tokens per slot" }
& $Server --model $ModelPath --alias $Alias --host 127.0.0.1 --port $Port --offline `
    --reasoning off --reasoning-budget 0 --gpu-layers $GpuLayers --fit off --ctx-size $Context --parallel $Parallel `
    --flash-attn on --no-webui --cors-origins "http://127.0.0.1:$Port" --log-verbosity 2
exit $LASTEXITCODE

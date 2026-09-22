# Build Windows x64 natives.
# Needs: Visual Studio 2022 (Desktop C++ workload), cmake, swig (`choco install swig`), a JDK.
#
# Output:
#   native\dist\resources\lightgbm4j\windows\x86_64\lib_lightgbm.dll{,.md5}
#   native\dist\resources\lightgbm4j\windows\x86_64\lib_lightgbm_swig.dll{,.md5}
#   native\dist\java\com\microsoft\ml\lightgbm\*.java
$ErrorActionPreference = "Stop"

$Native = $PSScriptRoot
$Src = Join-Path $Native "lightgbm"
$Build = Join-Path $Src "build"
$Dist = Join-Path $Native "dist"

if (-not (Test-Path (Join-Path $Src "external_libs\eigen\Eigen"))) {
  throw "LightGBM submodule missing: run 'git submodule update --init --recursive'"
}
foreach ($t in "cmake", "swig", "javac") {
  if (-not (Get-Command $t -ErrorAction SilentlyContinue)) { throw "$t not found on PATH" }
}
if (-not $env:JAVA_HOME) {
  $env:JAVA_HOME = (Get-Item (Get-Command javac).Source).Directory.Parent.FullName
}
Write-Host "building windows/x86_64 with JAVA_HOME=$env:JAVA_HOME"
swig -version

cmake -B $Build -S $Src -A x64 -DUSE_SWIG=ON -DBUILD_CLI=OFF
if ($LASTEXITCODE) { throw "cmake configure failed" }
cmake --build $Build --config Release -j $env:NUMBER_OF_PROCESSORS
if ($LASTEXITCODE) { throw "cmake build failed" }

# upstream's POST_BUILD step stages both libs here
$Stage = Join-Path $Build "com\microsoft\ml\lightgbm\windows\x86_64"
$Out = Join-Path $Dist "resources\lightgbm4j\windows\x86_64"
$JavaOut = Join-Path $Dist "java\com\microsoft\ml\lightgbm"
New-Item -ItemType Directory -Force $Out, $JavaOut | Out-Null
foreach ($lib in "lib_lightgbm", "lib_lightgbm_swig") {
  Copy-Item -Force (Join-Path $Stage "$lib.dll") (Join-Path $Out "$lib.dll")
  $md5 = (Get-FileHash -Algorithm MD5 (Join-Path $Out "$lib.dll")).Hash.ToLower()
  # 32 lowercase hex chars, no newline: LGBMBooster compares the sidecar as a string
  [IO.File]::WriteAllText((Join-Path $Out "$lib.dll.md5"), $md5)
}
Remove-Item -Force (Join-Path $JavaOut "*.java") -ErrorAction SilentlyContinue
Copy-Item (Join-Path $Build "java\*.java") $JavaOut

Get-ChildItem $Out

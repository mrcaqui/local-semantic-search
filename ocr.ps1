# ocr.ps1 — 段階3(代替案): Windows標準OCR(Windows.Media.Ocr)で画像内の文字を読む。追加インストール不要。
# Windows PowerShell 5.1 で実行する(PowerShell 7 では WinRT 型を直接読めない)。
#   powershell -ExecutionPolicy Bypass -File ocr.ps1 -ListFile data\images\sample.txt -OutFile data\images\ocr.jsonl
param(
  [Parameter(Mandatory)][string]$ListFile,   # 1行1画像。例: attachments/foo.png
  [Parameter(Mandatory)][string]$OutFile,
  [string]$VaultRoot = 'C:\Users\yomor\OneDrive\Desktop\vault',
  [string]$Lang = 'ja'
)
$ErrorActionPreference = 'Stop'
$outFull = [IO.Path]::GetFullPath($OutFile)
$vaultFull = [IO.Path]::GetFullPath($VaultRoot).TrimEnd('\') + '\'
if ($outFull.ToLower().StartsWith($vaultFull.ToLower())) { throw "出力先がVault内です: $outFull" }

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.FileAccessMode, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
  $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
} | Select-Object -First 1
function Await($op, [Type]$type) { $t = $asTask.MakeGenericMethod($type).Invoke($null, @($op)); $null = $t.Wait(-1); $t.Result }

$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new($Lang))
if ($null -eq $engine) { throw "OCR言語 '$Lang' が使えません。人間が日本語のOCR機能を追加してください。" }
$max = [Windows.Media.Ocr.OcrEngine]::MaxImageDimension

$out = New-Object System.Collections.Generic.List[string]
foreach ($rel in Get-Content -Encoding UTF8 $ListFile) {
  $rel = $rel.Trim().TrimStart([char]0xFEFF)
  if (-not $rel) { continue }
  $abs = [IO.Path]::GetFullPath((Join-Path $VaultRoot ($rel -replace '/', '\')))
  $stream = $null
  try {
    if (-not (Test-Path -LiteralPath $abs)) { throw "ファイルがありません: $abs" }
    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($abs)) ([Windows.Storage.StorageFile])
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    if ($decoder.PixelWidth -gt $max -or $decoder.PixelHeight -gt $max) { throw "画像が大きすぎます($($decoder.PixelWidth)x$($decoder.PixelHeight) > $max)" }
    $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
    $text = ($result.Lines | ForEach-Object { $_.Text }) -join "`n"
    $out.Add((@{ image = $rel; text = $text; width = $decoder.PixelWidth; height = $decoder.PixelHeight } | ConvertTo-Json -Compress))
  } catch {
    $ex = $_.Exception; while ($ex.InnerException) { $ex = $ex.InnerException }
    $out.Add((@{ image = $rel; text = ''; error = $ex.Message } | ConvertTo-Json -Compress))
  } finally {
    if ($stream) { $stream.Dispose() }
  }
}
[IO.File]::WriteAllLines($outFull, $out, (New-Object Text.UTF8Encoding $false))
Write-Output "OCR完了: $($out.Count) 件 → $outFull"

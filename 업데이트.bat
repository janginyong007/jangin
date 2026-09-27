@echo off
chcp 65001 >nul
echo ============================================
echo  캡컷 자동편집 프로그램 업데이트
echo  켜져 있는 프로그램을 끄고, 업데이트 후, 다시 켭니다.
echo ============================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; try { $c=@(Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue); foreach($x in $c){ Stop-Process -Id $x.OwningProcess -Force -ErrorAction SilentlyContinue; Write-Host 'Stopped running server (old version)' }; if($c.Count -gt 0){ Start-Sleep -Seconds 2 } } catch {}; $base='https://raw.githubusercontent.com/janginyong007/jangin/claude/capcut-auto-cut-detection-h0o5hh/src/'; $nc='?nocache=' + [DateTime]::Now.Ticks; $hits=@(Get-ChildItem -Path $env:USERPROFILE -Filter webapp.py -Recurse -Depth 4 -ErrorAction SilentlyContinue | Where-Object { $_.Directory.Name -eq 'src' -and (Test-Path (Join-Path $_.Directory.Parent.FullName 'config.yaml')) }); if($hits.Count -eq 0){ Write-Host 'ERROR: program folder not found'; exit 1 }; foreach($h in $hits){ $src=$h.DirectoryName; Write-Host ('Program folder: ' + $src); $bk=Join-Path $src ('update_backup_' + (Get-Date -Format yyyyMMdd_HHmmss)); New-Item -ItemType Directory -Path $bk | Out-Null; foreach($f in 'audio_energy.py','cutdetect.py','pipeline.py','transcribe.py'){ $p=Join-Path $src $f; if(Test-Path $p){ Copy-Item $p $bk }; Invoke-WebRequest -UseBasicParsing -Headers @{'Cache-Control'='no-cache'} -Uri ($base + $f + $nc) -OutFile $p; Write-Host ('  updated: ' + $f) }; $pc=Join-Path $src '__pycache__'; if(Test-Path $pc){ Remove-Item (Join-Path $pc '*') -Force -ErrorAction SilentlyContinue }; $v=(Select-String -Path (Join-Path $src 'pipeline.py') -Pattern '^VERSION' | Select-Object -First 1).Line; Write-Host ('  ' + $v); if(Select-String -Path (Join-Path $src 'pipeline.py') -Pattern 'write_cut_report' -Quiet){ Write-Host '  RESULT: SUCCESS' } else { Write-Host '  RESULT: FAILED'; exit 1 }; $root=$h.Directory.Parent.FullName; $launcher=Get-ChildItem -Path $root -Filter *.bat | Where-Object { (Get-Content $_.FullName -Raw) -match 'src\.webapp' } | Select-Object -First 1; if($launcher){ Write-Host ('Starting program: ' + $launcher.Name); try { Start-Process -FilePath $launcher.FullName -WorkingDirectory $root } catch { Write-Host 'Could not start automatically. Please start the program with your desktop icon.' } } else { Write-Host 'Please start the program with your desktop icon.' } }"
echo.
if errorlevel 1 (
  echo [실패] 위 메시지를 복사해서 Claude에게 보내주세요.
) else (
  echo [완료] 프로그램이 새로 켜졌습니다. 브라우저에서 다시 작업하세요.
  echo 로그 맨 위 [프로그램 버전] 줄이 위에 표시된 VERSION과 같으면 적용된 것입니다.
)
echo.
pause

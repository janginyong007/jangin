@echo off
chcp 65001 >nul
echo ============================================
echo  캡컷 자동편집 프로그램 업데이트
echo ============================================
echo.
:check
netstat -ano | findstr /R /C:"127\.0\.0\.1:8765  *0\.0\.0\.0:0" >nul
if not errorlevel 1 (
  echo [!] 캡컷 자동편집 프로그램이 아직 켜져 있습니다.
  echo     프로그램의 검은 창을 X 버튼으로 닫은 뒤, 이 창에서 아무 키나 누르세요.
  echo.
  pause >nul
  goto check
)
echo 업데이트 파일을 받는 중...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $base='https://raw.githubusercontent.com/janginyong007/jangin/claude/capcut-auto-cut-detection-h0o5hh/src/'; $nc='?nocache=' + [DateTime]::Now.Ticks; $hits=@(Get-ChildItem -Path $env:USERPROFILE -Filter webapp.py -Recurse -Depth 4 -ErrorAction SilentlyContinue | Where-Object { $_.Directory.Name -eq 'src' -and (Test-Path (Join-Path $_.Directory.Parent.FullName 'config.yaml')) }); if($hits.Count -eq 0){ Write-Host 'ERROR: program folder not found'; exit 1 }; foreach($h in $hits){ $src=$h.DirectoryName; Write-Host ('Program folder: ' + $src); $bk=Join-Path $src ('update_backup_' + (Get-Date -Format yyyyMMdd_HHmmss)); New-Item -ItemType Directory -Path $bk | Out-Null; foreach($f in 'audio_energy.py','cutdetect.py','pipeline.py','transcribe.py'){ $p=Join-Path $src $f; if(Test-Path $p){ Copy-Item $p $bk }; Invoke-WebRequest -UseBasicParsing -Uri ($base + $f + $nc) -OutFile $p; Write-Host ('  updated: ' + $f) }; $pc=Join-Path $src '__pycache__'; if(Test-Path $pc){ Remove-Item (Join-Path $pc '*') -Force -ErrorAction SilentlyContinue }; $v=(Select-String -Path (Join-Path $src 'pipeline.py') -Pattern '^VERSION' | Select-Object -First 1).Line; Write-Host ('  ' + $v); if(Select-String -Path (Join-Path $src 'pipeline.py') -Pattern 'write_cut_report' -Quiet){ Write-Host '  RESULT: SUCCESS' } else { Write-Host '  RESULT: FAILED'; exit 1 } }"
echo.
if errorlevel 1 (
  echo [실패] 위 메시지를 복사해서 Claude에게 보내주세요.
) else (
  echo [완료] 이제 바탕화면 아이콘으로 프로그램을 켜세요.
  echo 로그 맨 위 [프로그램 버전] 줄이 위의 VERSION과 같으면 적용된 것입니다.
)
echo.
pause

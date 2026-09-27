@echo off
chcp 65001 >nul
echo ============================================
echo  캡컷 자동편집 프로그램 업데이트
echo  (반복발화 / 말끝 잘림 / 컷리포트 수정판)
echo ============================================
echo.
echo 실행 중인 서버가 있으면 먼저 창을 닫아주세요.
echo 프로그램 폴더를 찾는 중...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $base='https://raw.githubusercontent.com/janginyong007/jangin/claude/capcut-auto-cut-detection-h0o5hh/src/'; $hits=@(Get-ChildItem -Path $env:USERPROFILE -Filter webapp.py -Recurse -Depth 4 -ErrorAction SilentlyContinue | Where-Object { $_.Directory.Name -eq 'src' -and (Test-Path (Join-Path $_.Directory.Parent.FullName 'config.yaml')) }); if($hits.Count -eq 0){ Write-Host 'ERROR: program folder not found'; exit 1 }; foreach($h in $hits){ $src=$h.DirectoryName; Write-Host ('Program folder: ' + $src); $bk=Join-Path $src ('update_backup_' + (Get-Date -Format yyyyMMdd_HHmmss)); New-Item -ItemType Directory -Path $bk | Out-Null; foreach($f in 'audio_energy.py','cutdetect.py','pipeline.py','transcribe.py'){ $p=Join-Path $src $f; if(Test-Path $p){ Copy-Item $p $bk }; Invoke-WebRequest -UseBasicParsing -Uri ($base + $f) -OutFile $p; Write-Host ('  updated: ' + $f) }; $pc=Join-Path $src '__pycache__'; if(Test-Path $pc){ Remove-Item (Join-Path $pc '*') -Force -ErrorAction SilentlyContinue }; if(Select-String -Path (Join-Path $src 'pipeline.py') -Pattern 'write_cut_report' -Quiet){ Write-Host '  RESULT: SUCCESS' } else { Write-Host '  RESULT: FAILED' } }"
echo.
if errorlevel 1 (
  echo [실패] 위 메시지를 복사해서 Claude에게 보내주세요.
) else (
  echo [완료] RESULT: SUCCESS 가 보이면 성공입니다.
  echo 이제 원래 쓰던 바탕화면 bat 파일로 프로그램을 실행하세요.
  echo 로그 맨 위에 [프로그램 버전] 2026-09-27 수정판 3 ... 줄이 보이면 적용된 것입니다.
)
echo.
pause

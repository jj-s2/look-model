$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

Set-Location 'c:\Users\John\Desktop\look model\third_party\mmaction2'

$py = 'c:\Users\John\Desktop\look model\.conda\envs\elderly-ai\python.exe'
$logFile = 'c:\Users\John\Desktop\look model\experiments\logs\mmaction_stgcn_demo.log'
$logDir = Split-Path $logFile -Parent
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$args = @(
    '-u',
    'demo\demo_skeleton.py',
    'demo\demo_skeleton.mp4',
    'demo\demo_skeleton_out_stgcn.mp4',
    '--config', 'configs\skeleton\stgcn\stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py',
    '--checkpoint', 'c:\Users\John\Desktop\look model\models\pretrained\stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.pth',
    '--det-config', 'demo\demo_configs\faster-rcnn_r50_fpn_2x_coco_infer.py',
    '--det-checkpoint', 'c:\Users\John\Desktop\look model\models\pretrained\faster_rcnn_r50_fpn_2x_coco.pth',
    '--det-score-thr', '0.9',
    '--det-cat-id', '0',
    '--pose-config', 'demo\demo_configs\td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
    '--pose-checkpoint', 'c:\Users\John\Desktop\look model\models\pretrained\hrnet_w32_coco_256x192-c78dce93_20200708.pth',
    '--label-map', 'tools\data\skeleton\label_map_ntu60.txt'
)

Write-Host "Starting STGCN demo at $(Get-Date -Format 'HH:mm:ss')"
Write-Host "Log file: $logFile"

& $py @args 2>&1 | Tee-Object -FilePath $logFile

Write-Host "Finished at $(Get-Date -Format 'HH:mm:ss') with exit code $LASTEXITCODE"

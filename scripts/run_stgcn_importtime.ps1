$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'

Set-Location 'c:\Users\John\Desktop\look model\third_party\mmaction2'

$py = 'c:\Users\John\Desktop\look model\.conda\envs\elderly-ai\python.exe'
$logFile = 'c:\Users\John\Desktop\look model\experiments\logs\stgcn_importtime.log'

$args = @(
    '-X', 'importtime',
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

Write-Host "Starting STGCN demo with importtime at $(Get-Date -Format 'HH:mm:ss')"

# -X importtime 输出到 stderr，单独重定向
& $py @args 1>$logFile 2>&1

Write-Host "Finished at $(Get-Date -Format 'HH:mm:ss') with exit code $LASTEXITCODE"

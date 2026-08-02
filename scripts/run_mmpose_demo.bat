@echo off
REM ============================================================
REM 复现 mmpose topdown demo (RTMDet-tiny + RTMPose-m)
REM 输出: vis_results/*.mp4 + vis_results/*.json
REM ============================================================
setlocal
set PY=C:\Users\John\Desktop\look model\.conda\envs\elderly-ai\python.exe
set ROOT=C:\Users\John\Desktop\look model\third_party\mmpose

cd /d "%ROOT%"

"%PY%" demo\topdown_demo_with_mmdet.py ^
    demo\mmdetection_cfg\rtmdet_tiny_8xb32-300e_coco.py ^
    https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth ^
    configs\body_2d_keypoint\rtmpose\coco\rtmpose-m_8xb256-420e_coco-256x192.py ^
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth ^
    --input demo\resources\demo.mp4 ^
    --output-root vis_results ^
    --save-predictions ^
    --device cuda:0

endlocal

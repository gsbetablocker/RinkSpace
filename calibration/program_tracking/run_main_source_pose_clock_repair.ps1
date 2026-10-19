$ErrorActionPreference = 'Stop'
$taskRoot = 'D:/OneDrive/codex_projs/RinkSpace'
$taskTmp = 'D:/CodexData/RinkSpace_program_tracking_tmp'
Set-Location -LiteralPath $taskRoot
& "$taskTmp/gtsam-env/python.exe" `
    "$taskRoot/three_frame_refit/program_tracking/fit_source_pose_clock_repaired.py" `
    --resume-source "$taskTmp/continuous_camera_model_v13_source_identity/main_long_side_seam_continue" `
    --pose-sidecar "$taskTmp/continuous_camera_model_v13_source_identity/source_image_pose_groups_v1.json" `
    --pose-sidecar-sha256 'c20a32a59c2141165bdc89b3b23619a7fb413e2b6a132f9f97b1f4bd689afa52' `
    --output "$taskTmp/continuous_camera_model_v14_source_pose_clock/main_long_side" `
    --iterations 80
exit $LASTEXITCODE

# Use only after the source-pose joint native/full-Jacobian gate passes.
$ErrorActionPreference = 'Stop'
$taskRoot = 'D:/OneDrive/codex_projs/RinkSpace'
$taskTmp = 'D:/CodexData/RinkSpace_program_tracking_tmp'
Set-Location -LiteralPath $taskRoot
& "$taskTmp/gtsam-env/python.exe" `
    "$taskRoot/three_frame_refit/program_tracking/fit_joint_shared_f2_programme.py" `
    --main-directory "$taskTmp/continuous_camera_model_v14_source_pose_clock/main_long_side" `
    --main-state accepted_states.npz `
    --low-directory "$taskTmp/continuous_camera_model_v11_time_principal/low_end" `
    --secondary-directory "$taskTmp/continuous_camera_model_v10_quadratic/secondary_long_side" `
    --output "$taskTmp/continuous_camera_model_v15_joint_f2" `
    --iterations 80
exit $LASTEXITCODE

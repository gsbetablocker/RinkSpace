# Prepared at user-requested pause. Run only when the user resumes the main task.
$ErrorActionPreference = 'Stop'
$taskRoot = 'D:/OneDrive/codex_projs/RinkSpace'
$taskTmp = 'D:/CodexData/RinkSpace_program_tracking_tmp'
$taskSource = "$taskTmp/continuous_camera_model_v15_joint_f2"
$taskOutput = "$taskTmp/continuous_camera_model_v15_joint_f2_resume_20260929"
if (Test-Path -LiteralPath $taskOutput) { throw 'Resume output exists; inspect it before continuing.' }
Set-Location -LiteralPath $taskRoot
& "$taskTmp/gtsam-env/python.exe" `
    "$taskRoot/three_frame_refit/program_tracking/fit_joint_shared_f2_programme.py" `
    --main-directory "$taskSource/frozen_main_input" `
    --main-state accepted_states.npz `
    --low-directory "$taskTmp/continuous_camera_model_v11_time_principal/low_end" `
    --secondary-directory "$taskTmp/continuous_camera_model_v10_quadratic/secondary_long_side" `
    --resume-joint $taskSource `
    --output $taskOutput `
    --iterations 80
exit $LASTEXITCODE

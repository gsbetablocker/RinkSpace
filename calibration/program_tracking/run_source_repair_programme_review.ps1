# Run after the main source-repair solver has exported its completed summary.
# This creates an explicitly unverified full-programme review, never a release.
$ErrorActionPreference = 'Stop'
$taskRoot = 'D:/OneDrive/codex_projs/RinkSpace'
$taskTmp = 'D:/CodexData/RinkSpace_program_tracking_tmp'
$taskPython = "$taskTmp/gtsam-env/python.exe"
$taskTracking = "$taskRoot/three_frame_refit/program_tracking"
$taskMain = "$taskTmp/continuous_camera_model_v14_source_pose_clock/main_long_side"
$taskSecondary = "$taskTmp/continuous_camera_model_v10_quadratic/secondary_long_side"
$taskLow = "$taskTmp/continuous_camera_model_v11_time_principal/low_end"
$taskReview = "$taskTmp/continuous_camera_model_v14_source_pose_clock/programme_review"
Set-Location -LiteralPath $taskRoot
foreach ($taskGroup in @($taskMain, $taskSecondary, $taskLow)) {
    if (-not (Test-Path -LiteralPath "$taskGroup/summary.json")) {
        throw "Completed camera export missing: $taskGroup"
    }
}
if (Test-Path -LiteralPath $taskReview) {
    throw 'Review output already exists; inspect its manifest before continuing.'
}
& $taskPython "$taskTracking/package_continuous_camera.py" `
    --group $taskMain --group $taskSecondary --group $taskLow --output $taskReview
if ($LASTEXITCODE -ne 0) { throw 'Programme camera packaging failed.' }
$taskPackage = Get-Content -LiteralPath "$taskReview/candidate_poses.json" -Raw | ConvertFrom-Json
if ($taskPackage.frames -ne 10140 -or $taskPackage.changed_frames -ne 10140) {
    throw 'All 10140 source frames must use the three explicit movable-camera groups.'
}
& $taskPython "$taskTracking/render_result.py" --candidate-review-video `
    --tracked "$taskReview/candidate_poses.npz" `
    --scene "$taskReview/candidate_scene.json" --release "$taskReview/candidate_release.json" `
    --render-temp "$taskReview/render_temp" `
    --output "$taskReview/okada_sp_source_pose_clock_grid_UNVERIFIED.mp4" `
    --cadence-qa "$taskReview/source_cadence_qa.json"
if ($LASTEXITCODE -ne 0) { throw 'Full programme review rendering failed; keep partial evidence.' }

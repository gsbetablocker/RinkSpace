# Render the exported complete joint camera stream, only after solve completion.
$ErrorActionPreference = 'Stop'
$taskRoot = 'D:/OneDrive/codex_projs/RinkSpace'
$taskTmp = 'D:/CodexData/RinkSpace_program_tracking_tmp'
$taskRun = "$taskTmp/continuous_camera_model_v15_joint_f2_resume_20260929"
$taskReview = "$taskRun/programme_review"
$taskForeground = "$taskTmp/continuous_camera_model_v2/secondary_visibility_source_review/foreground_source_mask/foreground_exclusion_manifest.json"
Set-Location -LiteralPath $taskRoot
if (-not (Test-Path -LiteralPath "$taskRun/summary.json")) {
    throw 'Joint solver has not completed its export.'
}
$taskSummary = Get-Content -LiteralPath "$taskRun/summary.json" -Raw | ConvertFrom-Json
$taskPackage = Get-Content -LiteralPath "$taskReview/candidate_poses.json" -Raw | ConvertFrom-Json
if ($taskPackage.frames -ne 10140 -or -not $taskPackage.complete_timeline -or
    $taskSummary.frames -ne 10140 -or -not $taskSummary.complete_timeline) {
    throw 'Joint export must preserve every original programme PTS.'
}
if ($taskSummary.joint_state_sha256 -ne (Get-FileHash -Algorithm SHA256 -LiteralPath "$taskRun/states.npz").Hash.ToLowerInvariant() -or
    $taskSummary.poses_sha256 -ne (Get-FileHash -Algorithm SHA256 -LiteralPath "$taskReview/candidate_poses.npz").Hash.ToLowerInvariant() -or
    $taskSummary.model_sha256 -ne (Get-FileHash -Algorithm SHA256 -LiteralPath "$taskReview/candidate_scene.json").Hash.ToLowerInvariant() -or
    $taskPackage.joint_state_sha256 -ne $taskSummary.joint_state_sha256 -or
    $taskPackage.poses_sha256 -ne $taskSummary.poses_sha256 -or
    $taskPackage.model_sha256 -ne $taskSummary.model_sha256) {
    throw 'Joint export summary, state, camera or shared scene hash differs.'
}
& "$taskTmp/gtsam-env/python.exe" "$taskRoot/three_frame_refit/program_tracking/render_with_geometry_runtime.py" `
    --candidate-review-video --tracked "$taskReview/candidate_poses.npz" `
    --scene "$taskReview/candidate_scene.json" --release "$taskReview/candidate_release.json" `
    --render-temp "$taskReview/render_temp" `
    --output "$taskReview/okada_sp_joint_f2_grid_UNVERIFIED.mp4" `
    --foreground-review "$taskForeground" `
    --cadence-qa "$taskReview/source_cadence_qa.json"
exit $LASTEXITCODE

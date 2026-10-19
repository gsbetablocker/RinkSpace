"use strict";

(() => {
  const $ = id => document.getElementById(id);
  const ui = Object.fromEntries([
    "pageTitle", "frameMeta", "previousFrame", "frameNumber", "jumpFrame", "nextFrame",
    "undoDraft", "redoDraft", "saveFrame", "saveAll", "draftCount", "status", "modeHelp",
    "viewport", "stage", "frameImage", "pointCanvas", "imageLoading", "imageView", "zoom",
    "cursorCoordinate", "pointCount", "draftState", "targetSearch", "targetSelect",
    "targetSummary", "pointDetail", "layerControls", "showLines", "showLabels", "showInferred",
    "undoSaved", "discardDraft", "conflictNotice", "sourceLimits"
  ].map(id => [id, $(id)]));
  const ctx = ui.pointCanvas.getContext("2d");
  const state = {
    bootstrap: null, indices: [], frames: new Map(), drafts: new Map(), conflicts: new Set(),
    current: null, mode: "box", selectedTarget: "", selectedPoint: null, visibleLayers: new Set(),
    busy: false, imageReady: false, width: 1920, height: 1080, box: null, request: 0,
    storageKey: null, storageFailed: false
  };
  const copy = value => JSON.parse(JSON.stringify(value));
  const sameRevision = (a, b) => String(a) === String(b);
  const currentFrame = () => state.frames.get(state.current);
  const currentDraft = () => state.drafts.get(state.current);
  const operations = index => state.drafts.get(index)?.operations || [];
  const hasDraft = index => operations(index).length > 0;
  const dirtyIndices = () => state.indices.filter(hasDraft);
  const kindLabels = {
    board_edge: "围挡边沿", ad_corner: "广告角点", ad_edge: "广告边沿",
    ad_print: "广告印刷轮廓", ice_curve: "冰面标线"
  };

  function status(message, type = "") {
    ui.status.textContent = message;
    ui.status.className = type;
  }

  async function api(url, body) {
    const opts = { cache: "no-store", headers: { Accept: "application/json" } };
    if (body !== undefined) {
      opts.method = "POST";
      opts.headers["Content-Type"] = "application/json";
      opts.headers["X-Editor-Token"] = state.bootstrap.csrf_token;
      opts.body = JSON.stringify(body);
    }
    let response;
    try { response = await fetch(url, opts); }
    catch (_) { throw new Error("无法连接本地编辑服务，草稿已保留。请确认服务仍在运行。"); }
    let result;
    try { result = await response.json(); }
    catch (_) { throw new Error(`编辑服务没有返回有效数据（HTTP ${response.status}），未确认保存。`); }
    if (!response.ok) {
      const detail = typeof result.error === "string" ? result.error : result.message || result.detail || "操作失败";
      const error = new Error(response.status === 409
        ? `此帧数据已更新，不能覆盖旧版本。${detail}`
        : `操作未完成：${detail}`);
      error.httpStatus = response.status;
      throw error;
    }
    return result;
  }

  function persistDrafts() {
    if (!state.storageKey) return;
    try {
      const entries = [...state.drafts].map(([index, draft]) => ({ index, ...draft }));
      localStorage.setItem(state.storageKey, JSON.stringify({ version: 1, entries }));
      state.storageFailed = false;
    } catch (_) {
      state.storageFailed = true;
      status("浏览器无法保存本机草稿，请先点击保存，再关闭页面。", "error");
    }
  }

  function restoreDrafts() {
    try {
      const text = localStorage.getItem(state.storageKey);
      if (!text) return;
      const saved = JSON.parse(text);
      if (saved.version !== 1 || !Array.isArray(saved.entries)) return;
      for (const entry of saved.entries) {
        if (!state.indices.includes(entry.index) || !Array.isArray(entry.operations)) continue;
        state.drafts.set(entry.index, {
          base_revision: entry.base_revision, operations: entry.operations,
          undo: Array.isArray(entry.undo) ? entry.undo : [], redo: Array.isArray(entry.redo) ? entry.redo : []
        });
      }
    } catch (_) { status("旧草稿无法读取。原始数据未改变。", "error"); }
  }

  function acceptFrame(frame) {
    state.frames.set(frame.index, frame);
    const draft = state.drafts.get(frame.index);
    if (draft && !sameRevision(draft.base_revision, frame.revision)) {
      if (draft.operations.length) state.conflicts.add(frame.index);
      else state.drafts.delete(frame.index);
    } else state.conflicts.delete(frame.index);
  }

  function frameSegments(frame = currentFrame(), includeDraft = true) {
    if (!frame) return [];
    const ops = includeDraft && !state.conflicts.has(frame.index) ? operations(frame.index) : [];
    const deleted = new Set(ops.filter(op => op.type === "delete").map(op => String(op.point_id)));
    const result = (frame.segments || []).flatMap(segment => {
      const points = (segment.points || []).map(point => ({
        ...point, id: String(point.id), target_id: point.target_id || segment.target_id
      }));
      if (!points.some(point => deleted.has(point.id))) return [{ ...segment, points }];
      const pieces = [];
      let piece = [];
      const finishPiece = () => {
        if (piece.length) {
          pieces.push({ ...segment, key: `${segment.key}/draft-split-${pieces.length}`, closed: false, points: piece, source_draw_uv: null });
          piece = [];
        }
      };
      for (const point of points) {
        if (deleted.has(point.id)) finishPiece();
        else piece.push(point);
      }
      finishPiece();
      return pieces;
    });
    for (const op of ops.filter(op => op.type === "add")) {
      const target = state.bootstrap.targets.find(item => item.id === op.target_id);
      result.push({
        key: op.local_id, feature_id: op.target_id, layer: target?.layer || "ice",
        target_id: op.target_id, closed: false,
        points: [{ id: op.local_id, target_id: op.target_id, uv: op.uv, added: true }]
      });
    }
    return result;
  }

  function allPoints(visibleOnly = false) {
    return frameSegments().flatMap(segment => {
      if (visibleOnly && !state.visibleLayers.has(segment.layer)) return [];
      return segment.points.map(point => ({ ...point, layer: segment.layer, segment_key: segment.key }));
    });
  }

  function layerColor(layer) { return state.bootstrap.layers[layer]?.[1] || "#dadfe8"; }

  function curvePath(uv, g) {
    if (uv.length < 2 || g?.kind !== "projected_physical_circle_ellipse") return uv;
    const [cx, cy] = g.center_uv, [a, b] = g.axes_px.map(v => v / 2);
    const alpha = g.angle_degrees * Math.PI / 180, c = Math.cos(alpha), s = Math.sin(alpha);
    const theta = uv.map(([x, y]) => Math.atan2((-(x-cx)*s+(y-cy)*c)/b, ((x-cx)*c+(y-cy)*s)/a));
    for (let i=1; i<theta.length; i++) {
      while (theta[i]-theta[i-1] > Math.PI) theta[i] -= 2*Math.PI;
      while (theta[i]-theta[i-1] < -Math.PI) theta[i] += 2*Math.PI;
    }
    const result = [];
    const add = t => result.push([cx+a*Math.cos(t)*c-b*Math.sin(t)*s, cy+a*Math.cos(t)*s+b*Math.sin(t)*c]);
    for (let i=1; i<theta.length; i++) {
      const steps = Math.max(1, Math.ceil(Math.abs(theta[i]-theta[i-1])*Math.max(a,b)/3));
      for (let j=0; j<steps; j++) add(theta[i-1]+(theta[i]-theta[i-1])*j/steps);
    }
    add(theta[theta.length-1]);
    return result;
  }

  function draw() {
    ctx.clearRect(0, 0, state.width, state.height);
    if (!currentFrame()) return;
    const scale = ui.pointCanvas.getBoundingClientRect().width / state.width || 1;
    for (const segment of frameSegments()) {
      if (!state.visibleLayers.has(segment.layer) || !segment.points.length) continue;
      const points = segment.points;
      const color = layerColor(segment.layer);
      if (ui.showLines.checked && points.length > 1) {
        const path = segment.source_draw_uv || curvePath(points.map(p => p.uv), segment.geometry);
        ctx.beginPath();
        ctx.moveTo(...path[0]);
        for (const point of path.slice(1)) ctx.lineTo(...point);
        if (segment.closed) ctx.closePath();
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.72;
        ctx.lineWidth = 1 / Math.max(scale, 0.5);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      for (const point of points) {
        ctx.beginPath();
        ctx.arc(point.uv[0], point.uv[1], Math.max(2.7, 2.3 / scale), 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.lineWidth = Math.max(0.8, 0.65 / scale);
        ctx.strokeStyle = "#10141c";
        ctx.stroke();
        if (point.added) {
          ctx.beginPath();
          ctx.arc(...point.uv, 5 / Math.max(scale, 0.6), 0, 2 * Math.PI);
          ctx.strokeStyle = color;
          ctx.lineWidth = 1 / Math.max(scale, 0.6);
          ctx.stroke();
        }
        if (point.id === state.selectedPoint) {
          ctx.beginPath();
          ctx.arc(...point.uv, 8 / scale, 0, 2 * Math.PI);
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1.8 / scale;
          ctx.stroke();
        }
      }
    }
    if (ui.showLabels.checked) {
      const labels = new Map();
      for (const segment of frameSegments()) {
        if (!state.visibleLayers.has(segment.layer) || !segment.points.length) continue;
        const target = state.bootstrap.targets.find(item => item.id === segment.target_id);
        let label = target?.label || segment.label || segment.target_id;
        if (label.startsWith("Ad")) label = label.split(" ")[0];
        if (target?.kind === "board_edge") label = label.split(" · ")[0];
        label = displayLabel(label);
        const previous = labels.get(label);
        if (!previous || segment.points.length > previous.points.length) labels.set(label, segment);
      }
      ctx.font = `${13 / scale}px sans-serif`;
      ctx.textBaseline = "bottom";
      const occupied = [];
      for (const [label, segment] of labels) {
        const point = segment.points[Math.floor(segment.points.length / 2)].uv;
        const width = ctx.measureText(label).width, height = 16 / scale;
        const x = Math.max(3 / scale, Math.min(state.width - width - 3 / scale, point[0] + 6 / scale));
        let y = Math.max(height, Math.min(state.height - 3 / scale, point[1] - 5 / scale));
        for (let attempt = 0; attempt < 8 && occupied.some(r => x < r.x + r.w && x + width > r.x && y > r.y - height && y - height < r.y); attempt++) {
          y = y + height < state.height ? y + height : Math.max(height, y - 2 * height);
        }
        occupied.push({x, y, w: width});
        ctx.lineWidth = 3 / scale;
        ctx.strokeStyle = "rgba(0,0,0,.7)";
        ctx.strokeText(label, x, y);
        ctx.fillStyle = layerColor(segment.layer);
        ctx.fillText(label, x, y);
      }
    }
    if (ui.showInferred.checked) {
      for (const segment of currentFrame().inferred_segments || []) {
        if (segment.layer && !state.visibleLayers.has(segment.layer)) continue;
        const points = segment.points || segment.uv || segment.path_uv || segment.source_polyline || [];
        const uv = points.map(point => Array.isArray(point) ? point : point.uv).filter(point => Array.isArray(point));
        if (uv.length < 2) continue;
        ctx.beginPath();
        ctx.moveTo(...uv[0]);
        uv.slice(1).forEach(point => ctx.lineTo(...point));
        ctx.strokeStyle = layerColor(segment.layer);
        ctx.setLineDash([7 / scale, 5 / scale]);
        ctx.lineWidth = 1 / scale;
        ctx.globalAlpha = 0.6;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
    }
    if (state.box) {
      const { start, end } = state.box;
      ctx.fillStyle = "rgba(255,99,75,.13)";
      ctx.strokeStyle = "#ffad96";
      ctx.lineWidth = 1.5 / scale;
      ctx.fillRect(start[0], start[1], end[0] - start[0], end[1] - start[1]);
      ctx.strokeRect(start[0], start[1], end[0] - start[0], end[1] - start[1]);
    }
  }

  function updateControls() {
    const frame = currentFrame();
    const draft = currentDraft();
    const conflict = state.conflicts.has(state.current);
    const disabled = state.busy || !frame;
    const dirty = dirtyIndices();
    const position = state.indices.indexOf(state.current);
    ui.previousFrame.disabled = disabled || position <= 0;
    ui.nextFrame.disabled = disabled || position >= state.indices.length - 1;
    ui.jumpFrame.disabled = state.busy;
    ui.undoDraft.disabled = disabled || conflict || !draft?.undo.length;
    ui.redoDraft.disabled = disabled || conflict || !draft?.redo.length;
    ui.saveFrame.disabled = disabled || frame?.read_only || conflict || !hasDraft(state.current);
    ui.saveAll.disabled = disabled || !dirty.length;
    ui.discardDraft.disabled = disabled || !draft;
    ui.undoSaved.disabled = disabled || frame?.read_only || hasDraft(state.current) || !frame?.history_available;
    ui.draftCount.textContent = dirty.length ? `(${dirty.length})` : "";
    ui.conflictNotice.hidden = !conflict;
    document.querySelectorAll("[data-mode]").forEach(button => {
      button.classList.toggle("active", button.dataset.mode === state.mode);
      button.disabled = disabled || (frame?.read_only && button.dataset.mode !== "inspect");
    });
    if (frame) {
      const count = allPoints().length;
      const savedCount = frame.source_point_count ?? frame.point_count;
      ui.pointCount.textContent = `${count} / ${state.bootstrap.hard_point_ceiling || 500} 个点${count !== savedCount ? `（已保存 ${savedCount}）` : ""}`;
      const deletes = operations(state.current).filter(op => op.type === "delete").length;
      const adds = operations(state.current).filter(op => op.type === "add").length;
      ui.draftState.textContent = frame.read_only ? "已由用户验收 · 原数据只读" : conflict ? "旧草稿冲突，未应用"
        : hasDraft(state.current) ? `尚未保存：删 ${deletes}、加 ${adds}` : "本帧没有待保存修改";
    }
    updateModeHelp();
  }

  function updateModeHelp() {
    if (currentFrame()?.read_only) { ui.modeHelp.textContent = "本帧已验收。点击标点查看坐标与模型归属。"; return; }
    const help = {
      box: "拖出矩形，松手删除框内显示的点。误删可按 Ctrl+Z 撤销。",
      delete: "点击一个点就删除。小点周围也可点击；误删可按 Ctrl+Z 撤销。",
      add: state.selectedTarget
        ? `点击原图加点，归属：${targetName(state.selectedTarget)}。新增点单独登记。`
        : "先在右侧选择模型标准物，再点击原图加点。",
      inspect: "点击已有点，查看其坐标与模型标准物；选好后可切换“加点”。"
    };
    ui.modeHelp.textContent = ui.imageView.value === "navigation"
      ? "当前为 8a28 网格参考。可查看标准物，编辑请切换回原图。" : help[state.mode];
    ui.pointCanvas.style.cursor = state.mode === "inspect" ? "pointer" : "crosshair";
  }

  function displayLabel(label) { return label.replace(/\s*·\s*/g, " "); }

  function targetName(id) { return displayLabel(state.bootstrap.targets.find(target => target.id === id)?.label || id); }

  function targetSearchText(value) { return value.toLowerCase().replace(/[\s._·-]+/g, ""); }

  function compareTargets(a, b) {
    const roles = ["left_top", "left_bottom", "right_top", "right_bottom", "top_edge", "bottom_edge", "left_vertical", "right_vertical"];
    const key = target => {
      const ad = /^Ad(\d+)\.(.+)$/.exec(target.id);
      if (ad) {
        const kind = { ad_corner: 0, ad_edge: 1, ad_print: 2 }[target.kind] ?? 3;
        const role = roles.indexOf(ad[2]);
        return [2, Number(ad[1]), kind, role < 0 ? roles.length : role];
      }
      return [target.kind === "board_edge" ? 0 : target.kind === "ice_curve" ? 1 : 3, 0, 0, 0];
    };
    const ka = key(a), kb = key(b);
    for (let i = 0; i < ka.length; i++) if (ka[i] !== kb[i]) return ka[i] - kb[i];
    return a.id.localeCompare(b.id, "en", { numeric: true });
  }

  function renderTargets() {
    const search = targetSearchText(ui.targetSearch.value.trim());
    const selected = state.selectedTarget;
    const matching = state.bootstrap.targets.filter(target =>
      !search || targetSearchText(`${target.id} ${target.label} ${displayLabel(target.label || target.id)}`).includes(search) || target.id === selected);
    matching.sort(compareTargets);
    ui.targetSelect.replaceChildren();
    for (const target of matching) {
      const option = document.createElement("option");
      option.value = target.id;
      option.textContent = displayLabel(target.label || target.id);
      option.title = target.id;
      ui.targetSelect.append(option);
    }
    ui.targetSelect.value = selected;
  }

  function selectTarget(id) {
    const target = state.bootstrap.targets.find(item => item.id === id);
    state.selectedTarget = target?.id || "";
    renderTargets();
    ui.targetSummary.textContent = target
      ? `${targetName(target.id)}\n${target.id} · ${kindLabels[target.kind] || target.kind}`
      : "尚未选择标准物。";
    updateModeHelp();
  }

  function inspectPoint(point) {
    state.selectedPoint = point?.id || null;
    if (point) {
      if (point.unbound) {
        ui.pointDetail.textContent = `像素 (${point.uv[0].toFixed(2)}, ${point.uv[1].toFixed(2)})\n${point.label}\n实体边界已读取，标识物身份待确认；未进入场地拟合。`;
        status("正在查看身份待确认的冰面实体边界。");
        draw();
        return;
      }
      selectTarget(point.target_id);
      ui.pointDetail.textContent = `像素 (${point.uv[0].toFixed(2)}, ${point.uv[1].toFixed(2)})\n标准物：${targetName(point.target_id)}\n点编号：${point.id}${point.added ? "（新加，尚未保存）" : ""}`;
      status(`已选取标准物：${targetName(point.target_id)}`);
    } else ui.pointDetail.textContent = "点击已有点可查看坐标与归属。";
    draw();
  }

  function scaleStage() {
    let scale = Number(ui.zoom.value);
    if (ui.zoom.value === "fit") {
      scale = Math.min(ui.viewport.clientWidth / state.width, ui.viewport.clientHeight / state.height);
    }
    if (!(scale > 0)) return;
    ui.stage.style.width = `${state.width * scale}px`;
    ui.stage.style.height = `${state.height * scale}px`;
    draw();
  }

  function loadImage(frame) {
    state.imageReady = false;
    ui.imageLoading.hidden = false;
    ui.imageLoading.textContent = "正在读取画面…";
    const url = ui.imageView.value === "source" ? frame.source_url : frame.navigation_url;
    ui.frameImage.alt = `第 ${frame.index} 帧${ui.imageView.value === "source" ? "原图" : "8a28 网格参考"}`;
    ui.frameImage.onload = () => {
      state.width = ui.frameImage.naturalWidth;
      state.height = ui.frameImage.naturalHeight;
      ui.pointCanvas.width = state.width;
      ui.pointCanvas.height = state.height;
      state.imageReady = true;
      ui.imageLoading.hidden = true;
      scaleStage();
    };
    ui.frameImage.onerror = () => {
      state.imageReady = false;
      ui.imageLoading.textContent = "画面读取失败，请重试或检查本地服务。";
      status("画面读取失败，当前不能编辑。", "error");
    };
    ui.frameImage.src = url;
    draw();
  }

  async function goFrame(index) {
    if (state.busy || !state.indices.includes(index)) return;
    const request = ++state.request;
    state.busy = true;
    updateControls();
    status(`正在读取第 ${index} 帧…`, "busy");
    try {
      const frame = await api(`/api/frames/${index}`);
      if (request !== state.request) return;
      acceptFrame(frame);
      state.current = index;
      state.selectedPoint = null;
      state.box = null;
      ui.frameNumber.value = index;
      ui.frameMeta.textContent = `第 ${state.indices.indexOf(index) + 1}/${state.indices.length} 张 · 节目第 ${index} 帧 · 源时间 ${frame.source_time} · PTS ${frame.pts}`;
      ui.sourceLimits.textContent = (frame.source_limits || []).map(region => {
        const roi = region.source_roi_xyxy || region.roi_xyxy;
        return `${region.note_zh || region.reason || region.note || "材料边界暂不能可靠区分"}${roi ? `（原图范围 ${roi.join(", ")}）` : ""}`;
      }).join("\n\n") || "未另列未可靠读取区域；所有标点均待本次逐张人工审核。";
      ui.pointDetail.textContent = "点击已有点可查看坐标与归属。";
      loadImage(frame);
      history.replaceState(null, "", `#frame=${index}`);
      if (state.conflicts.has(index)) status("旧草稿与当前数据冲突。丢弃此帧草稿后可重新编辑。", "error");
      else status(frame.read_only ? `第 ${index} 帧已验收，可查看标点及模型归属。` : hasDraft(index) ? `已恢复第 ${index} 帧草稿，尚未保存到数据集。` : `第 ${index} 帧已读取，可直接编辑。`);
    } catch (error) { status(error.message, "error"); }
    finally { state.busy = false; updateControls(); }
  }

  function canEdit() {
    if (state.busy || !currentFrame() || !state.imageReady) return false;
    if (currentFrame().read_only) { status("这一帧已验收，当前页面只读。"); return false; }
    if (ui.imageView.value !== "source") {
      status("请切换到原图后编辑；8a28 网格用于判断标准物位置。");
      return false;
    }
    if (state.conflicts.has(state.current)) {
      status("旧草稿不能覆盖当前数据，请先丢弃此帧草稿。", "error");
      return false;
    }
    return true;
  }

  function mutate(nextOperations, message) {
    if (!canEdit()) return;
    let draft = currentDraft();
    if (!draft) {
      draft = { base_revision: currentFrame().revision, operations: [], undo: [], redo: [] };
      state.drafts.set(state.current, draft);
    }
    if (JSON.stringify(nextOperations) === JSON.stringify(draft.operations)) return;
    draft.undo.push(copy(draft.operations));
    draft.operations = nextOperations;
    draft.redo = [];
    state.selectedPoint = null;
    ui.pointDetail.textContent = "点击已有点可查看坐标与归属。";
    persistDrafts();
    updateControls();
    draw();
    if (!state.storageFailed) status(`${message}；尚未保存，可撤销。`);
  }

  function deletePoints(points) {
    points = points.filter(point => !point.read_only);
    if (!points.length) { status("没有选中点。隐藏层的点不会被框选删除。"); return; }
    const ids = new Set(points.map(point => point.id));
    const next = copy(operations(state.current)).filter(op => !(op.type === "add" && ids.has(op.local_id)));
    const deleted = new Set(next.filter(op => op.type === "delete").map(op => String(op.point_id)));
    for (const point of points) {
      if (!point.added && !deleted.has(point.id)) next.push({ type: "delete", point_id: point.id });
    }
    mutate(next, `已从草稿删除 ${points.length} 个点`);
  }

  function undoLocal(redo = false) {
    if (state.busy || state.conflicts.has(state.current)) return;
    const draft = currentDraft();
    const source = redo ? draft?.redo : draft?.undo;
    if (!source?.length) return;
    (redo ? draft.undo : draft.redo).push(copy(draft.operations));
    draft.operations = source.pop();
    state.selectedPoint = null;
    persistDrafts();
    updateControls();
    draw();
    if (!state.storageFailed) status(redo ? "已重做本帧草稿操作。" : "已撤销本帧草稿操作。");
  }

  async function saveIndices(indices) {
    if (state.busy || !indices.length) return;
    state.busy = true;
    updateControls();
    let saved = 0;
    let savingIndex = null;
    try {
      for (const index of indices) {
        savingIndex = index;
        if (state.conflicts.has(index)) throw new Error(`第 ${index} 帧的旧草稿存在冲突，未保存。`);
        const draft = state.drafts.get(index);
        if (!draft?.operations.length) continue;
        status(`正在保存第 ${index} 帧（${saved + 1}/${indices.length}）…`, "busy");
        const payload = draft.operations.map(op => op.type === "add"
          ? { type: "add", target_id: op.target_id, uv: op.uv }
          : { type: "delete", point_id: op.point_id });
        const frame = await api(`/api/frames/${index}/edit`, { base_revision: draft.base_revision, operations: payload });
        state.drafts.delete(index);
        state.conflicts.delete(index);
        acceptFrame(frame);
        saved++;
        persistDrafts();
        if (index === state.current) { state.selectedPoint = null; draw(); }
      }
      if (!state.storageFailed) status(`已保存 ${saved} 帧，修改与标准物登记已写入数据集。`, "success");
    } catch (error) {
      if (error.httpStatus === 409 && savingIndex !== null) {
        state.conflicts.add(savingIndex);
        try { acceptFrame(await api(`/api/frames/${savingIndex}`)); } catch (_) { /* Keep the conflicting draft. */ }
      }
      status(`${saved ? `前 ${saved} 帧已保存。` : ""}${error.message} 剩余草稿已保留。`, "error");
      draw();
    } finally { state.busy = false; updateControls(); }
  }

  function eventUV(event) {
    const rect = ui.pointCanvas.getBoundingClientRect();
    return [
      Math.max(0, Math.min(state.width - 1, (event.clientX - rect.left) * state.width / rect.width)),
      Math.max(0, Math.min(state.height - 1, (event.clientY - rect.top) * state.height / rect.height))
    ];
  }

  function nearestPoint(uv) {
    const scale = ui.pointCanvas.getBoundingClientRect().width / state.width;
    let minimum = (9 / scale) ** 2;
    let found = null;
    for (const point of allPoints(true)) {
      const distance = (point.uv[0] - uv[0]) ** 2 + (point.uv[1] - uv[1]) ** 2;
      if (distance <= minimum) { minimum = distance; found = point; }
    }
    return found;
  }

  ui.pointCanvas.addEventListener("pointerdown", event => {
    if (event.button !== 0 || state.busy || !state.imageReady) return;
    const uv = eventUV(event);
    if (state.mode === "inspect" || currentFrame()?.read_only) { inspectPoint(nearestPoint(uv)); return; }
    if (!canEdit()) return;
    if (state.mode === "delete") { const point = nearestPoint(uv); if (point) deletePoints([point]); else status("没有选中点，请点击点附近。"); }
    else if (state.mode === "add") {
      if (!state.selectedTarget) { status("请先在右侧选择模型标准物，或切到“查看 / 选标准物”点击已有点。", "error"); return; }
      const ceiling = state.bootstrap.hard_point_ceiling || 500;
      if (allPoints().length >= ceiling) { status(`本帧已达到 ${ceiling} 点硬上限，请先删点。`, "error"); return; }
      const target = state.bootstrap.targets.find(item => item.id === state.selectedTarget);
      state.visibleLayers.add(target.layer);
      const layerInput = ui.layerControls.querySelector(`[data-layer="${CSS.escape(target.layer)}"]`);
      if (layerInput) layerInput.checked = true;
      const op = { type: "add", target_id: target.id, uv: uv.map(value => Math.round(value * 10000) / 10000), local_id: `draft-${crypto.randomUUID()}` };
      mutate([...copy(operations(state.current)), op], `已加 1 点，归属 ${targetName(target.id)}`);
    } else if (state.mode === "box") {
      state.box = { start: uv, end: uv, pointer: event.pointerId };
      ui.pointCanvas.setPointerCapture(event.pointerId);
      draw();
    }
  });

  ui.pointCanvas.addEventListener("pointermove", event => {
    const uv = eventUV(event);
    ui.cursorCoordinate.textContent = `原图坐标：${uv[0].toFixed(1)}, ${uv[1].toFixed(1)}`;
    if (!state.box || state.box.pointer !== event.pointerId) return;
    state.box.end = uv;
    const { start, end } = state.box;
    const count = allPoints(true).filter(point => point.uv[0] >= Math.min(start[0], end[0]) && point.uv[0] <= Math.max(start[0], end[0])
      && point.uv[1] >= Math.min(start[1], end[1]) && point.uv[1] <= Math.max(start[1], end[1])).length;
    ui.modeHelp.textContent = `框内 ${count} 个点。松手删除；按 Esc 取消。`;
    draw();
  });

  ui.pointCanvas.addEventListener("pointerup", event => {
    if (!state.box || state.box.pointer !== event.pointerId) return;
    const { start } = state.box;
    const end = eventUV(event);
    state.box = null;
    if (ui.pointCanvas.hasPointerCapture(event.pointerId)) ui.pointCanvas.releasePointerCapture(event.pointerId);
    const scale = ui.pointCanvas.getBoundingClientRect().width / state.width;
    if (Math.hypot(end[0] - start[0], end[1] - start[1]) * scale < 4) inspectPoint(nearestPoint(end));
    else deletePoints(allPoints(true).filter(point => point.uv[0] >= Math.min(start[0], end[0]) && point.uv[0] <= Math.max(start[0], end[0])
      && point.uv[1] >= Math.min(start[1], end[1]) && point.uv[1] <= Math.max(start[1], end[1])));
    updateModeHelp();
    draw();
  });
  ui.pointCanvas.addEventListener("pointercancel", () => { state.box = null; updateModeHelp(); draw(); });
  ui.pointCanvas.addEventListener("mouseleave", () => { if (!state.box) ui.cursorCoordinate.textContent = "原图坐标：—"; });

  function setMode(mode) { state.mode = mode; state.box = null; updateControls(); draw(); }
  document.querySelectorAll("[data-mode]").forEach(button => button.addEventListener("click", () => setMode(button.dataset.mode)));
  ui.previousFrame.addEventListener("click", () => goFrame(state.indices[state.indices.indexOf(state.current) - 1]));
  ui.nextFrame.addEventListener("click", () => goFrame(state.indices[state.indices.indexOf(state.current) + 1]));
  ui.jumpFrame.addEventListener("click", () => {
    const index = Number(ui.frameNumber.value);
    if (state.indices.includes(index)) goFrame(index);
    else status(`请输入 ${state.indices[0]}–${state.indices.at(-1)} 之间的帧号。`, "error");
  });
  ui.frameNumber.addEventListener("keydown", event => { if (event.key === "Enter") ui.jumpFrame.click(); });
  ui.imageView.addEventListener("change", () => { state.box = null; if (currentFrame()) loadImage(currentFrame()); updateModeHelp(); });
  ui.zoom.addEventListener("change", scaleStage);
  new ResizeObserver(scaleStage).observe(ui.viewport);
  ui.showLines.addEventListener("change", draw);
  ui.showLabels.addEventListener("change", draw);
  ui.showInferred.addEventListener("change", draw);
  ui.targetSearch.addEventListener("input", renderTargets);
  ui.targetSelect.addEventListener("change", () => selectTarget(ui.targetSelect.value));
  ui.undoDraft.addEventListener("click", () => undoLocal());
  ui.redoDraft.addEventListener("click", () => undoLocal(true));
  ui.saveFrame.addEventListener("click", () => saveIndices([state.current]));
  ui.saveAll.addEventListener("click", () => saveIndices(dirtyIndices()));
  ui.discardDraft.addEventListener("click", () => {
    if (state.busy) return;
    const wasConflict = state.conflicts.has(state.current);
    state.drafts.delete(state.current);
    state.conflicts.delete(state.current);
    state.selectedPoint = null;
    persistDrafts();
    updateControls();
    draw();
    if (wasConflict) goFrame(state.current);
    else if (!state.storageFailed) status("已丢弃此帧草稿，显示当前已保存的点。");
  });
  ui.undoSaved.addEventListener("click", async () => {
    if (state.busy || hasDraft(state.current) || !currentFrame()?.history_available) return;
    state.busy = true;
    updateControls();
    status("正在撤销本帧上次保存…", "busy");
    try {
      const frame = await api(`/api/frames/${state.current}/undo`, { base_revision: currentFrame().revision });
      state.drafts.delete(state.current);
      state.conflicts.delete(state.current);
      acceptFrame(frame);
      state.selectedPoint = null;
      persistDrafts();
      draw();
      status("已撤销本帧上次保存，恢复后的点与登记已写入数据集。", "success");
    } catch (error) { status(error.message, "error"); }
    finally { state.busy = false; updateControls(); }
  });

  window.addEventListener("keydown", event => {
    const key = event.key.toLowerCase();
    const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName) || event.target.isContentEditable;
    if ((event.ctrlKey || event.metaKey) && key === "s") { event.preventDefault(); if (hasDraft(state.current)) saveIndices([state.current]); return; }
    if (typing) return;
    if ((event.ctrlKey || event.metaKey) && (key === "z" || key === "y")) {
      event.preventDefault(); undoLocal(key === "y" || event.shiftKey); return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey || state.busy) return;
    if (key === "escape") { state.box = null; state.selectedPoint = null; updateModeHelp(); draw(); }
    else if (key === "arrowleft") { event.preventDefault(); ui.previousFrame.click(); }
    else if (key === "arrowright") { event.preventDefault(); ui.nextFrame.click(); }
    else if ({ b: "box", d: "delete", a: "add", i: "inspect" }[key]) setMode({ b: "box", d: "delete", a: "add", i: "inspect" }[key]);
    else if (key === "delete" && state.selectedPoint) {
      const point = allPoints().find(item => item.id === state.selectedPoint);
      if (point) deletePoints([point]);
    }
  });
  window.addEventListener("beforeunload", event => {
    if (dirtyIndices().length || state.busy) { event.preventDefault(); event.returnValue = ""; }
  });

  async function start() {
    try {
      state.bootstrap = await api("/api/bootstrap");
      state.indices = state.bootstrap.frame_indices;
      state.storageKey = `rinkspace.point-editor.v1:${state.bootstrap.dataset_id || state.bootstrap.title}:${state.indices[0]}-${state.indices.at(-1)}`;
      state.visibleLayers = new Set(Object.keys(state.bootstrap.layers));
      document.title = ui.pageTitle.textContent = state.bootstrap.title;
      ui.frameNumber.min = state.indices[0];
      ui.frameNumber.max = state.indices.at(-1);
      for (const [layer, [label, color]] of Object.entries(state.bootstrap.layers)) {
        const container = document.createElement("label");
        container.className = "check";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = true;
        checkbox.dataset.layer = layer;
        checkbox.addEventListener("change", () => { if (checkbox.checked) state.visibleLayers.add(layer); else state.visibleLayers.delete(layer); draw(); });
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.backgroundColor = color;
        container.append(checkbox, swatch, document.createTextNode(label));
        ui.layerControls.append(container);
      }
      restoreDrafts();
      renderTargets();
      const requested = Number(new URLSearchParams(location.hash.slice(1)).get("frame"));
      await goFrame(state.indices.includes(requested) ? requested : state.indices[0]);
    } catch (error) { status(`页面初始化失败：${error.message}`, "error"); }
  }
  start();
})();

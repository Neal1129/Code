(function () {
    "use strict";
    var state = {
        currentTaskId: null,
        loadedTaskId: null,
        isUploading: false,
        pollingTimer: null,
        pollInFlight: false,
        historyTasks: [],
        historyCollapsed: false,
        cmpIds: [],
        cmpPayloads: {},
        cmpRows: []
    };
    var PARAM_FIELDS = [
        { key: "cannyLow", slider: "cannyLow", input: "cannyLowInput", type: "int", apiKey: "CANNY_LOW" },
        { key: "cannyHigh", slider: "cannyHigh", input: "cannyHighInput", type: "int", apiKey: "CANNY_HIGH" },
        { key: "houghDP", slider: "houghDP", input: "houghDPInput", type: "float", apiKey: "HOUGH_DP" },
        { key: "houghMinDist", slider: "houghMinDist", input: "houghMinDistInput", type: "int", apiKey: "HOUGH_MIN_DIST" },
        { key: "houghParam1", slider: "houghParam1", input: "houghParam1Input", type: "int", apiKey: "HOUGH_PARAM1" },
        { key: "houghParam2", slider: "houghParam2", input: "houghParam2Input", type: "int", apiKey: "HOUGH_PARAM2" },
        { key: "minRadius", slider: "minRadius", input: "minRadiusInput", type: "int", apiKey: "MIN_RADIUS" },
        { key: "maxRadius", slider: "maxRadius", input: "maxRadiusInput", type: "int", apiKey: "MAX_RADIUS" },
        { key: "trackMaxMiss", slider: "trackMaxMiss", input: "trackMaxMissInput", type: "int", apiKey: "TRACK_MAX_MISS" },
        { key: "matchDistance", slider: "matchDistance", input: "matchDistanceInput", type: "int", apiKey: "MATCH_DISTANCE" },
        { key: "pixelToMicron", slider: "pixelToMicron", input: "pixelToMicronInput", type: "float", apiKey: "PIXEL_TO_MICRON" },
        { key: "speedMin", slider: "speedMin", input: "speedMinInput", type: "int", apiKey: "speed_min" },
        { key: "interfaceRatio", slider: "interfaceRatio", input: "interfaceRatioInput", type: "float", apiKey: "INTERFACE_RATIO" },
        { key: "chipLength", slider: "chipLength", input: "chipLengthInput", type: "int", apiKey: "CHIP_LENGTH_MICRON" },
        { key: "targetThreshold", slider: "targetThreshold", input: "targetThresholdInput", type: "int", apiKey: "TARGET_THRESHOLD_MICRON" }
    ];
    var CMP_METRICS = [
        { key: "penetration", label: "直线率", fmt: fmtRate, dfmt: fmtPPDelta, scale: "ratio" },
        { key: "avg_directionality", label: "方向性", fmt: fmtNum3, dfmt: fmtSgn3, scale: "ratio" },
        { key: "trapped", label: "困陷率", fmt: fmtRate, dfmt: fmtPPDelta, scale: "ratio" },
        { key: "avg_displacement", label: "平均位移(μm)", fmt: fmtNum2um, dfmt: fmtSgn2um, scale: "dynamic" },
        { key: "first_time", label: "首到达时间(s)", fmt: fmtNum2s, dfmt: fmtSgn2s, scale: "dynamic" },
        { key: "target_rate", label: "目标到达率", fmt: fmtRate, dfmt: fmtPPDelta, scale: "ratio" },
        { key: "avg_speed", label: "均速(μm/s)", fmt: fmtNum2v, dfmt: fmtSgn2v, scale: "dynamic" }
    ];
    var CHART_IDS = {
        speedChart: "speed",
        displacementChart: "displacement",
        directionalityChart: "directionality",
        trapChart: "trap",
        fptChart: "fpt",
        ratesChart: "rates",
        classificationChart: "classification"
    };
    function el(id) {
        return document.getElementById(id);
    }
    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }
    function fmtRate(value) {
        var number = +value;
        return (isFinite(number) ? clamp(number, 0, 1) * 100 : 0).toFixed(1) + "%";
    }
    function fmtNum(value, digits) {
        var number = +value;
        return isFinite(number) ? number.toFixed(digits) : "0.00";
    }
    function fmtNum2(value) {
        return fmtNum(value, 2);
    }
    function fmtNum3(value) {
        return fmtNum(value, 3);
    }
    function fmtNum2um(value) {
        return fmtNum2(value) + "μm";
    }
    function fmtNum2s(value) {
        return fmtNum2(value) + "s";
    }
    function fmtNum2v(value) {
        return fmtNum2(value) + "μm/s";
    }
    function fmtSigned(value, digits) {
        var number = +value;
        if (!isFinite(number)) {
            return "+0.00";
        }
        return (number >= 0 ? "+" : "") + number.toFixed(digits);
    }
    function fmtSgn3(value) {
        return fmtSigned(value, 3);
    }
    function fmtSgn2um(value) {
        return fmtSigned(value, 2) + "μm";
    }
    function fmtSgn2s(value) {
        return fmtSigned(value, 2) + "s";
    }
    function fmtSgn2v(value) {
        return fmtSigned(value, 2) + "μm/s";
    }
    function fmtPPDelta(value) {
        var number = +value * 100;
        if (!isFinite(number)) {
            return "+0.0pp";
        }
        return (number >= 0 ? "+" : "") + number.toFixed(1) + "pp";
    }
    function fmtDuration(value) {
        var number = +value;
        if (!isFinite(number) || number <= 0) {
            return "--";
        }
        if (number < 60) {
            return number.toFixed(1) + "s";
        }
        return Math.floor(number / 60) + "m " + (number % 60).toFixed(0) + "s";
    }
    function fmtDateTime(value) {
        var number = +value;
        if (!isFinite(number) || number <= 0) {
            return "--";
        }
        return new Date(number * 1000).toLocaleString("zh-CN", { hour12: false });
    }
    function cacheBust(url) {
        return url ? url + (url.indexOf("?") === -1 ? "?" : "&") + "t=" + Date.now() : "";
    }
    function normTags(value) {
        var source = Array.isArray(value) ? value : String(value || "").split(",");
        var tags = [];
        var seen = {};
        source.forEach(function (item) {
            var tag = String(item || "").trim();
            var lowered = tag.toLowerCase();
            if (!tag || seen[lowered] || tags.length >= 8) {
                return;
            }
            seen[lowered] = true;
            tags.push(tag);
        });
        return tags;
    }
    function parseValue(value, type) {
        return type === "float" ? parseFloat(value) : parseInt(value, 10);
    }
    function apiFetch(url, options) {
        return fetch(url, options || {}).then(function (response) {
            return response.text().then(function (text) {
                var data = {};
                if (text) {
                    try {
                        data = JSON.parse(text);
                    } catch (error) {
                        throw new Error("后端返回了非JSON响应");
                    }
                }
                if (!response.ok) {
                    throw new Error(data.error || ("请求失败: HTTP " + response.status));
                }
                return data;
            });
        });
    }
    function setText(id, value) {
        var node = el(id);
        if (node) {
            node.textContent = value;
        }
    }
    function setDisplay(id, visible) {
        var node = el(id);
        if (node) {
            node.style.display = visible ? "block" : "none";
        }
    }
    function setWidth(id, value) {
        var node = el(id);
        if (node) {
            node.style.width = clamp(+value || 0, 0, 100) + "%";
        }
    }
    function setStatus(text) {
        var node = el("statusText");
        if (!node) {
            return;
        }
        node.style.display = text ? "block" : "none";
        if (text) {
            node.textContent = text;
        }
    }
    function showProgress(visible) {
        var node = el("processingStatus");
        if (node) {
            node.style.display = visible ? "block" : "none";
        }
    }
    function setProgressBar(percent, stageText, isPaused) {
        var button = el("pauseResumeBtn");
        setWidth("progressBarFill", percent);
        setText("progressPct", Math.round(+percent || 0) + "%");
        setText("progressStageText", stageText || "正在处理视频...");
        if (button) {
            button.textContent = isPaused ? "▶ 继续处理" : "⏸ 暂停处理";
            button.className = "btn progress-pause-btn" + (isPaused ? " paused" : "");
        }
    }
    function setTab(name) {
        document.querySelectorAll(".tab-content").forEach(function (node) {
            node.classList.remove("active");
        });
        document.querySelectorAll(".tab-btn").forEach(function (node) {
            node.classList.toggle("active", (node.getAttribute("onclick") || "").indexOf("'" + name + "'") !== -1);
        });
        var tab = el(name + "-tab");
        if (tab) {
            tab.classList.add("active");
        }
    }
    function switchTab(event, name) {
        setTab(name);
        if (event && event.currentTarget) {
            event.currentTarget.classList.add("active");
        }
    }
    function collectParams() {
        var params = {};
        PARAM_FIELDS.forEach(function (field) {
            var slider = el(field.slider);
            params[field.key] = slider ? parseValue(slider.value, field.type) : 0;
        });
        return params;
    }
    function syncInputs(params) {
        PARAM_FIELDS.forEach(function (field) {
            var slider = el(field.slider);
            var input = el(field.input);
            if (slider) {
                slider.value = params[field.key];
            }
            if (input) {
                input.value = params[field.key];
            }
        });
    }
    function buildParamPayload(params) {
        var payload = {};
        PARAM_FIELDS.forEach(function (field) {
            payload[field.apiKey] = params[field.key];
        });
        return payload;
    }
    function updateParams() {
        var params = collectParams();
        syncInputs(params);
        apiFetch("/update_params", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(buildParamPayload(params))
        }).catch(function (error) {
            console.error("参数更新失败:", error);
        });
    }
    function updateParamFromInput(sliderId, inputId, type) {
        var slider = el(sliderId);
        var input = el(inputId);
        if (!slider || !input) {
            return;
        }
        slider.value = parseValue(input.value, type);
        updateParams();
    }
    function updatePlayButton() {
        var video = el("resultVideo");
        setText("playPauseBtn", video && video.paused === false ? "⏸ 暂停" : "▶ 播放");
    }
    function bindVideoEvents() {
        var video = el("resultVideo");
        if (!video || video.__bound__) {
            return;
        }
        video.__bound__ = true;
        ["play", "pause", "ended"].forEach(function (eventName) {
            video.addEventListener(eventName, updatePlayButton);
        });
        video.addEventListener("error", function () {
            var fallbackUrl = video.dataset.fallbackUrl || "";
            if (fallbackUrl && video.dataset.fallbackTried !== "1") {
                video.dataset.fallbackTried = "1";
                video.src = cacheBust(fallbackUrl);
                video.load();
                return;
            }
            setStatus("视频播放失败。请刷新页面；若仍无法播放，可从结果目录打开下载版视频。");
        });
    }
    function attachVideo(url) {
        var video = el("resultVideo");
        if (!video) {
            return;
        }
        bindVideoEvents();
        video.pause();
        if (!url) {
            video.removeAttribute("src");
            delete video.dataset.fallbackUrl;
            delete video.dataset.fallbackTried;
            video.load();
            updatePlayButton();
            return;
        }
        var playbackUrl = url;
        if (/tracking_clustered\.mp4(?:\?|$)/i.test(url)) {
            playbackUrl = url.replace(/tracking_clustered\.mp4/i, "tracking_clustered.webm");
            video.dataset.fallbackUrl = url;
        } else {
            delete video.dataset.fallbackUrl;
        }
        video.dataset.fallbackTried = "0";
        video.src = cacheBust(playbackUrl);
        video.setAttribute("type", playbackUrl.toLowerCase().indexOf(".webm") !== -1 ? "video/webm" : "video/mp4");
        video.load();
        video.oncanplay = function () {
            video.play().catch(function () {});
            video.oncanplay = null;
        };
        updatePlayButton();
    }
    function togglePlayPause() {
        var video = el("resultVideo");
        if (!video) {
            return;
        }
        if (!video.src || video.src === window.location.href) {
            setStatus("请先等待视频处理完成");
            return;
        }
        if (video.paused) {
            video.play().then(updatePlayButton).catch(function () {
                setStatus("视频暂时无法播放");
                updatePlayButton();
            });
            return;
        }
        video.pause();
        updatePlayButton();
    }
    function resetResults() {
        attachVideo(null);
        setDisplay("analysisResults", false);
        setDisplay("noResults", true);
        setDisplay("chartsAnalysis", false);
        setDisplay("noChartsResults", true);
        ["penetrationRate", "penetrationCount", "totalTracks", "targetRate", "targetCount", "eligibleTracks", "trappedRate", "directionality", "firstArrivalTime", "avgDisplacement", "outputVideoPath", "outputJsonPath", "outputCsvPath", "outputTrackMetricsPath", "outputChartsPath"].forEach(function (id) {
            setText(id, "");
        });
        ["penetrationBar", "targetBar", "trappedBar"].forEach(function (id) {
            setWidth(id, 0);
        });
    }
    function updateChartImages(charts) {
        Object.keys(CHART_IDS).forEach(function (id) {
            var node = el(id);
            var url = charts[CHART_IDS[id]];
            if (node && url) {
                node.src = cacheBust(url);
            }
        });
    }
    function applyResults(payload) {
        var metrics = payload.metrics || {};
        var paths = payload.paths || {};
        var task = payload.task || {};
        state.loadedTaskId = payload.task_id || null;
        state.currentTaskId = payload.task_id || null;
        setText("penetrationRate", fmtRate(metrics.penetration));
        setText("targetRate", fmtRate(metrics.target_rate));
        setText("trappedRate", fmtRate(metrics.trapped));
        setText("directionality", isFinite(+metrics.avg_directionality) ? (+metrics.avg_directionality).toFixed(3) : "0");
        setText("firstArrivalTime", isFinite(+metrics.first_time) ? (+metrics.first_time).toFixed(2) + "s" : "0s");
        setText("avgDisplacement", isFinite(+metrics.avg_displacement) ? (+metrics.avg_displacement).toFixed(2) + "μm" : "0μm");
        setText("penetrationCount", metrics.straight_tracks || 0);
        setText("totalTracks", metrics.quality_eligible_tracks || 0);
        setText("targetCount", metrics.target_tracks || 0);
        setText("eligibleTracks", metrics.target_eligible_tracks || 0);
        setWidth("penetrationBar", (+metrics.penetration || 0) * 100);
        setWidth("targetBar", (+metrics.target_rate || 0) * 100);
        setWidth("trappedBar", (+metrics.trapped || 0) * 100);
        attachVideo(payload.video_url);
        updateChartImages(payload.charts || {});
        setText("outputVideoPath", paths.video || "");
        setText("outputJsonPath", paths.json || "");
        setText("outputCsvPath", paths.csv || "");
        setText("outputTrackMetricsPath", paths.track_metrics_csv || "");
        setText("outputChartsPath", paths.charts_dir || "");
        setDisplay("analysisResults", true);
        setDisplay("noResults", false);
        setDisplay("chartsAnalysis", true);
        setDisplay("noChartsResults", false);
        if (task.original_filename) {
            setStatus("已加载任务: " + task.original_filename);
        }
        setTab("results");
        renderHistoryList();
    }
    function stopPolling() {
        if (state.pollingTimer) {
            clearInterval(state.pollingTimer);
            state.pollingTimer = null;
        }
    }
    function fetchTaskResult(taskId) {
        return apiFetch("/tasks/" + taskId + "/result").then(function (result) {
            state.cmpPayloads[taskId] = result;
            return result;
        });
    }
    function startPolling(taskId) {
        stopPolling();
        doPoll(taskId);
        state.pollingTimer = setInterval(function () {
            doPoll(taskId);
        }, 1000);
    }
    function doPoll(taskId) {
        if (!taskId || state.pollInFlight) {
            return;
        }
        state.pollInFlight = true;
        apiFetch("/tasks/" + taskId + "/status").then(function (data) {
            if (data.status === "queued" || data.status === "processing") {
                var message = data.progress_message || data.message || "正在处理视频...";
                setStatus(data.is_paused ? "已暂停" : message);
                showProgress(true);
                setProgressBar(data.progress || 0, message, !!data.is_paused);
                return null;
            }
            if (data.status === "done") {
                stopPolling();
                return fetchTaskResult(taskId).then(function (result) {
                    applyResults(result);
                    showProgress(false);
                    setProgressBar(100, "处理完成", false);
                    setStatus("处理完成！");
                    return refreshTaskHistory();
                });
            }
            if (data.status === "failed") {
                stopPolling();
                showProgress(false);
                setStatus("处理失败: " + (data.error || "").split("\n")[0]);
                return null;
            }
            setStatus("状态未知，重试中...");
            return null;
        }).catch(function (error) {
            console.error("轮询状态失败:", error);
            setStatus("查询状态失败，重试中...");
        }).then(function () {
            state.pollInFlight = false;
        });
    }
    function upload() {
        var input = el("videoInput");
        var file = input && input.files && input.files[0];
        if (!file) {
            alert("请先选择视频文件");
            return;
        }
        state.isUploading = true;
        state.currentTaskId = null;
        stopPolling();
        resetResults();
        setStatus("正在上传视频...");
        showProgress(true);
        setProgressBar(0, "准备上传...", false);
        var formData = new FormData();
        formData.append("video", file);
        apiFetch("/upload", { method: "POST", body: formData }).then(function (data) {
            state.currentTaskId = data.task_id;
            setStatus(data.message || "上传成功，正在处理...");
            setProgressBar(0, "已上传，等待开始...", false);
            startPolling(data.task_id);
        }).catch(function (error) {
            console.error("上传失败:", error);
            showProgress(false);
            setStatus("上传失败: " + error.message);
        }).then(function () {
            state.isUploading = false;
        });
    }
    function togglePauseResume() {
        var taskId = state.currentTaskId;
        var button = el("pauseResumeBtn");
        if (!taskId || !button) {
            return;
        }
        apiFetch("/tasks/" + taskId + "/" + (button.classList.contains("paused") ? "resume" : "pause"), { method: "POST" }).catch(function (error) {
            console.error("暂停/继续失败:", error);
            setStatus("操作失败: " + error.message);
        });
    }
    function updateHistoryPanel() {
        var layout = el("appLayout");
        var toggleButton = el("historyToggleBtn");
        var edgeButton = el("historyEdgeToggle");
        if (!layout || !toggleButton || !edgeButton) {
            return;
        }
        layout.classList.toggle("history-collapsed", state.historyCollapsed);
        toggleButton.textContent = state.historyCollapsed ? "展开" : "收起";
        edgeButton.style.display = state.historyCollapsed ? "flex" : "none";
    }
    function toggleHistoryPanel() {
        state.historyCollapsed = !state.historyCollapsed;
        updateHistoryPanel();
    }
    function createNode(tag, className, text) {
        var node = document.createElement(tag);
        if (className) {
            node.className = className;
        }
        if (text !== undefined) {
            node.textContent = text;
        }
        return node;
    }
    function createSummaryChips(summary) {
        var wrapper = createNode("div", "history-card-metrics");
        ["穿透率 " + fmtRate(summary.penetration_rate || 0), "到达率 " + fmtRate(summary.target_arrival_rate || 0), "均速 " + fmtNum2(summary.average_speed || 0) + "μm/s"].forEach(function (text) {
            wrapper.appendChild(createNode("span", "history-metric-chip", text));
        });
        return wrapper;
    }
    function createTagChips(tags) {
        var wrapper = createNode("div", "history-card-tags");
        tags.forEach(function (tag) {
            wrapper.appendChild(createNode("span", "history-tag-chip", tag));
        });
        return wrapper;
    }
    function updateCardSelection() {
        state.cmpIds = state.cmpIds.filter(function (id) {
            return state.historyTasks.some(function (task) {
                return task.task_id === id;
            });
        });
    }
    function renderHistoryList() {
        var list = el("historyList");
        var empty = el("historyEmpty");
        if (!list || !empty) {
            return;
        }
        list.innerHTML = "";
        updateCardSelection();
        if (!state.historyTasks.length) {
            empty.style.display = "block";
            return;
        }
        empty.style.display = "none";
        state.historyTasks.forEach(function (task) {
            var tags = normTags(task.tags || []);
            var card = createNode("div", "history-card" + (state.loadedTaskId === task.task_id ? " active" : "") + (state.cmpIds.indexOf(task.task_id) !== -1 ? " compare-selected" : ""));
            var top = createNode("div", "history-card-top");
            var actions = createNode("div", "history-card-actions");
            var name = createNode("div", "history-card-name", task.original_filename || "未知原文件名");
            var time = createNode("div", "history-card-time");
            var meta = createNode("div", "history-card-meta");
            var noteInput = createNode("textarea", "history-note-input");
            var tagInput = createNode("input", "history-tag-input");
            var saveButton = createNode("button", "history-meta-save", "保存备注/标签");
            var starButton = createNode("button", "history-card-star" + (task.starred ? " active" : ""), task.starred ? "★ 已星标" : "☆ 星标");
            var compareButton = createNode("button", "history-card-compare" + (state.cmpIds.indexOf(task.task_id) !== -1 ? " selected" : ""), state.cmpIds.indexOf(task.task_id) !== -1 ? "已选对比" : "加入对比");
            time.innerHTML = "处理时间: " + fmtDateTime(task.finished_at || task.created_at) + "<br>耗时: " + fmtDuration(task.processing_seconds);
            noteInput.rows = 2;
            noteInput.placeholder = "添加任务备注，例如：培养基A组，第1次重复";
            noteInput.value = task.note || "";
            tagInput.type = "text";
            tagInput.placeholder = "标签，用英文逗号分隔";
            tagInput.value = tags.join(", ");
            card.addEventListener("click", function () {
                loadHistoricalTask(task.task_id);
            });
            starButton.type = "button";
            compareButton.type = "button";
            saveButton.type = "button";
            starButton.addEventListener("click", function (event) {
                event.stopPropagation();
                saveTaskMeta(task.task_id, task.note || "", task.tags || [], !task.starred);
            });
            compareButton.addEventListener("click", function (event) {
                event.stopPropagation();
                toggleCmp(task.task_id);
            });
            meta.addEventListener("click", function (event) {
                event.stopPropagation();
            });
            saveButton.addEventListener("click", function (event) {
                event.stopPropagation();
                saveButton.disabled = true;
                saveButton.textContent = "保存中...";
                saveTaskMeta(task.task_id, noteInput.value, tagInput.value, !!task.starred).catch(function (error) {
                    setStatus("保存失败: " + error.message);
                }).then(function () {
                    saveButton.disabled = false;
                    saveButton.textContent = "保存备注/标签";
                });
            });
            top.appendChild(createNode("div", "history-card-id", "#" + task.serial_number));
            actions.appendChild(starButton);
            actions.appendChild(compareButton);
            top.appendChild(actions);
            meta.appendChild(noteInput);
            meta.appendChild(tagInput);
            meta.appendChild(saveButton);
            card.appendChild(top);
            card.appendChild(name);
            card.appendChild(time);
            card.appendChild(createSummaryChips(task.summary || {}));
            if (tags.length) {
                card.appendChild(createTagChips(tags));
            }
            card.appendChild(meta);
            list.appendChild(card);
        });
    }
    function refreshTaskHistory() {
        return apiFetch("/tasks/history").then(function (data) {
            state.historyTasks = Array.isArray(data.tasks) ? data.tasks : [];
            renderHistoryList();
            return renderComparison();
        }).catch(function (error) {
            console.error("刷新历史失败:", error);
            setStatus("加载历史任务失败: " + error.message);
        });
    }
    function loadHistoricalTask(taskId) {
        state.loadedTaskId = taskId;
        renderHistoryList();
        setStatus("正在加载历史任务...");
        fetchTaskResult(taskId).then(function (result) {
            applyResults(result);
        }).catch(function (error) {
            console.error("加载历史任务失败:", error);
            state.loadedTaskId = null;
            renderHistoryList();
            setStatus("加载历史任务失败: " + error.message);
        });
    }
    function saveTaskMeta(taskId, note, tags, starred) {
        return apiFetch("/tasks/" + taskId + "/meta", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ note: String(note || "").trim(), tags: normTags(tags), starred: !!starred })
        }).then(function (saved) {
            state.historyTasks = state.historyTasks.map(function (task) {
                return task.task_id === taskId ? Object.assign({}, task, { note: saved.note, tags: saved.tags, starred: saved.starred }) : task;
            });
            renderHistoryList();
            setStatus("任务备注已保存");
        });
    }
    function toggleCmp(taskId) {
        var index = state.cmpIds.indexOf(taskId);
        if (index !== -1) {
            state.cmpIds.splice(index, 1);
        } else {
            if (state.cmpIds.length >= 2) {
                state.cmpIds.shift();
            }
            state.cmpIds.push(taskId);
        }
        renderHistoryList();
        renderComparison();
    }
    function clearTaskComparison() {
        state.cmpIds = [];
        state.cmpRows = [];
        renderHistoryList();
        renderComparison();
    }
    function getMetric(payload, key) {
        var metrics = (payload && payload.metrics) || {};
        var value = +metrics[key];
        return isFinite(value) ? value : 0;
    }
    function renderCmpTable(payloadA, payloadB) {
        var body = el("comparisonTableBody");
        if (!body) {
            return;
        }
        body.innerHTML = "";
        state.cmpRows = [];
        CMP_METRICS.forEach(function (metric) {
            var valueA = getMetric(payloadA, metric.key);
            var valueB = getMetric(payloadB, metric.key);
            var delta = valueB - valueA;
            var row = document.createElement("tr");
            row.innerHTML = "<td>" + metric.label + "</td><td>" + metric.fmt(valueA) + "</td><td>" + metric.fmt(valueB) + "</td><td class='" + (delta > 0 ? "delta-positive" : delta < 0 ? "delta-negative" : "") + "'>" + metric.dfmt(delta) + "</td>";
            body.appendChild(row);
            state.cmpRows.push({ label: metric.label, a: metric.fmt(valueA), b: metric.fmt(valueB), delta: metric.dfmt(delta) });
        });
    }
    function drawSeries(ctx, payload, radius, count, centerX, centerY, maxMap, stroke, fill) {
        ctx.beginPath();
        CMP_METRICS.forEach(function (metric, index) {
            var angle = Math.PI * 2 * index / count - Math.PI / 2;
            var value = clamp(getMetric(payload, metric.key) / (maxMap[metric.key] || 1), 0, 1);
            var x = centerX + Math.cos(angle) * radius * value;
            var y = centerY + Math.sin(angle) * radius * value;
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 2;
        ctx.fill();
        ctx.stroke();
    }
    function drawRadar(payloadA, payloadB) {
        var canvas = el("comparisonRadar");
        if (!canvas) {
            return;
        }
        var ctx = canvas.getContext("2d");
        var width = canvas.width;
        var height = canvas.height;
        var centerX = width / 2;
        var centerY = height / 2 + 20;
        var radius = Math.min(width, height) * 0.28;
        var count = CMP_METRICS.length;
        var maxMap = {};
        CMP_METRICS.forEach(function (metric) {
            if (metric.scale === "ratio") {
                maxMap[metric.key] = 1;
            } else {
                maxMap[metric.key] = Math.max(Math.abs(getMetric(payloadA, metric.key)), Math.abs(getMetric(payloadB, metric.key)), 0.001);
            }
        });
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, width, height);
        for (var ring = 1; ring <= 5; ring += 1) {
            ctx.beginPath();
            for (var i = 0; i < count; i += 1) {
                var ringAngle = Math.PI * 2 * i / count - Math.PI / 2;
                var ringRadius = radius * ring / 5;
                var ringX = centerX + Math.cos(ringAngle) * ringRadius;
                var ringY = centerY + Math.sin(ringAngle) * ringRadius;
                if (i === 0) {
                    ctx.moveTo(ringX, ringY);
                } else {
                    ctx.lineTo(ringX, ringY);
                }
            }
            ctx.closePath();
            ctx.strokeStyle = "#cbd5e1";
            ctx.lineWidth = 1;
            ctx.stroke();
        }
        CMP_METRICS.forEach(function (metric, index) {
            var angle = Math.PI * 2 * index / count - Math.PI / 2;
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius);
            ctx.strokeStyle = "#94a3b8";
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = "#334155";
            ctx.font = "13px Arial";
            ctx.textAlign = centerX + Math.cos(angle) * (radius + 30) >= centerX ? "left" : "right";
            ctx.fillText(metric.label, centerX + Math.cos(angle) * (radius + 30), centerY + Math.sin(angle) * (radius + 18));
        });
        drawSeries(ctx, payloadA, radius, count, centerX, centerY, maxMap, "#2563eb", "rgba(37,99,235,0.18)");
        drawSeries(ctx, payloadB, radius, count, centerX, centerY, maxMap, "#dc2626", "rgba(220,38,38,0.18)");
        ctx.font = "13px Arial";
        ctx.textAlign = "left";
        ctx.fillStyle = "#2563eb";
        ctx.fillRect(20, 20, 14, 14);
        ctx.fillStyle = "#334155";
        ctx.fillText("任务 A", 40, 32);
        ctx.fillStyle = "#dc2626";
        ctx.fillRect(90, 20, 14, 14);
        ctx.fillStyle = "#334155";
        ctx.fillText("任务 B", 110, 32);
    }
    function renderComparison() {
        var visible = state.cmpIds.length === 2;
        setDisplay("comparisonEmpty", !visible);
        setDisplay("comparisonContent", visible);
        if (!visible) {
            return Promise.resolve();
        }
        return Promise.all(state.cmpIds.map(function (taskId) {
            return state.cmpPayloads[taskId] ? Promise.resolve(state.cmpPayloads[taskId]) : fetchTaskResult(taskId);
        })).then(function (results) {
            var taskA = results[0].task || {};
            var taskB = results[1].task || {};
            setText("compareTaskALabel", "#" + state.cmpIds[0].slice(0, 8) + " " + (taskA.original_filename || "").slice(0, 14));
            setText("compareTaskBLabel", "#" + state.cmpIds[1].slice(0, 8) + " " + (taskB.original_filename || "").slice(0, 14));
            setText("comparisonTableHeadA", "#" + state.cmpIds[0].slice(0, 8));
            setText("comparisonTableHeadB", "#" + state.cmpIds[1].slice(0, 8));
            renderCmpTable(results[0], results[1]);
            drawRadar(results[0], results[1]);
        }).catch(function (error) {
            console.error("对比加载失败:", error);
            setStatus("对比加载失败: " + error.message);
        });
    }
    function downloadBlob(name, blob) {
        var link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = name;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    function exportComparisonCsv() {
        if (!state.cmpRows.length) {
            alert("请先选择两个历史任务进行对比");
            return;
        }
        var lines = ["指标,任务A,任务B,差值"];
        state.cmpRows.forEach(function (row) {
            lines.push([row.label, row.a, row.b, row.delta].join(","));
        });
        downloadBlob("comparison_" + Date.now() + ".csv", new Blob(["\uFEFF" + lines.join("\n")], { type: "text/csv;charset=utf-8" }));
    }
    function exportComparisonChart() {
        var canvas = el("comparisonRadar");
        if (state.cmpIds.length < 2 || !canvas) {
            alert("请先选择两个历史任务进行对比");
            return;
        }
        var link = document.createElement("a");
        link.href = canvas.toDataURL("image/png");
        link.download = "comparison_radar_" + Date.now() + ".png";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    window.switchTab = switchTab;
    window.updateParams = updateParams;
    window.updateParamFromInput = updateParamFromInput;
    window.upload = upload;
    window.togglePauseResume = togglePauseResume;
    window.togglePlayPause = togglePlayPause;
    window.toggleHistoryPanel = toggleHistoryPanel;
    window.refreshTaskHistory = refreshTaskHistory;
    window.clearTaskComparison = clearTaskComparison;
    window.exportComparisonCsv = exportComparisonCsv;
    window.exportComparisonChart = exportComparisonChart;
    window.addEventListener("DOMContentLoaded", function () {
        var input = el("videoInput");
        bindVideoEvents();
        updateParams();
        updateHistoryPanel();
        if (input) {
            input.addEventListener("change", function () {
                if (this.files && this.files[0]) {
                    upload();
                }
            });
        }
        refreshTaskHistory();
    });
})();

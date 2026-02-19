document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadContent = document.getElementById('upload-content');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');

    const ageInput = document.getElementById('age');
    const heightInput = document.getElementById('height');
    const weightInput = document.getElementById('weight');

    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingState = document.getElementById('loading-state');
    const resultDisplay = document.getElementById('result-display');
    const errorBox = document.getElementById('error-box');

    let selectedFile = null;

    // --- State Management ---
    function checkValidity() {
        const isValid = selectedFile &&
            ageInput.value &&
            heightInput.value &&
            weightInput.value;

        // Only enable if not loading
        if (!loadingState.classList.contains('hidden')) {
            analyzeBtn.disabled = true;
        } else {
            analyzeBtn.disabled = !isValid;
        }
    }

    [ageInput, heightInput, weightInput].forEach(inp => {
        inp.addEventListener('input', checkValidity);
    });

    // --- File Handling ---
    dropZone.addEventListener('click', (e) => {
        if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', function () {
        handleFiles(this.files);
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            // Relaxed validation: just check if it's an image
            if (!file.type.match('image.*')) {
                showError("Please upload a valid image file.");
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                uploadContent.classList.add('hidden');
                previewContainer.classList.remove('hidden');
            };
            reader.readAsDataURL(file);

            selectedFile = file;
            clearError();
            // 7. Keep previous result: Do NOT hide resultDisplay here
            // resultDisplay.classList.add('hidden'); 
            checkValidity();
        }
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = '';
        uploadContent.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        imagePreview.src = '';
        analyzeBtn.disabled = true;
        // Optionally keep result or clear it? User said "Replace only when new valid result arrives"
        // But if image is removed, maybe it implies clearing? 
        // Let's keep it visible until new analysis for maximum safety/persistence.
    });

    // --- Analysis Logic ---
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI Updates: Loading Mode
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...'; // Feedback
        loadingState.classList.remove('hidden');
        errorBox.classList.add('hidden');
        disableInputs(true);

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('age', ageInput.value);
        formData.append('height', heightInput.value);
        formData.append('weight', weightInput.value);

        try {
            // 1. Correct Fetch Usage
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            // 2. Safe JSON Parsing
            const data = await response.json().catch(() => null);

            if (!data) {
                throw new Error("Invalid response from server.");
            }

            if (data.success) {
                displayResult(data);
            } else {
                showError(data.error || "Analysis failed.");
            }

        } catch (error) {
            console.error(error);
            showError("Connection failed. Please check your internet or server status.");
        } finally {
            // Restore UI
            loadingState.classList.add('hidden');
            analyzeBtn.textContent = 'Analyze Lesion';
            disableInputs(false);
            checkValidity(); // Re-enable button if still valid
        }
    });

    function disableInputs(disabled) {
        ageInput.disabled = disabled;
        heightInput.disabled = disabled;
        weightInput.disabled = disabled;
        dropZone.style.pointerEvents = disabled ? 'none' : 'auto';
    }

    function displayResult(data) {
        // Only now replace the content
        resultDisplay.innerHTML = '';

        // Animation reset
        resultDisplay.classList.remove('pop-in');
        void resultDisplay.offsetWidth;
        resultDisplay.classList.add('pop-in');

        // Ensure stageId is present (default to 0 if missing/undefined)
        const stageId = (data.stage_id !== undefined) ? data.stage_id : 0;

        // Heatmap
        let heatmapHtml = '';
        if (data.heatmap_image) {
            heatmapHtml = `
                <div class="heatmap-section" style="margin-top: 30px; text-align: center;">
                    <h4 class="metric-label" style="margin-bottom: 15px;">Lesion Analysis Heatmap</h4>
                    <div class="heatmap-container" style="display: inline-block; padding: 5px; background: white; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                         <img src="data:image/jpeg;base64,${data.heatmap_image}" alt="Lesion Analysis Heatmap" style="border-radius: 8px; max-width: 100%;">
                    </div>
                </div>
            `;
        }

        const html = `
            <div class="medical-report-card stage-${stageId}">
                <div class="report-content">
                    <!-- Header Restored (Title & Date Only) -->
                    <div class="report-header">
                        <div class="report-title">MEDICAL ANALYSIS REPORT</div>
                        <div class="report-id">${new Date().toLocaleDateString(undefined, { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</div>
                    </div>

                    <!-- Final Diagnosis -->
                    <div class="diagnosis-section">
                        <div class="diagnosis-label">Final Diagnosis</div>
                        <h1 class="diagnosis-value">${data.diagnosis}</h1>
                        
                        <div class="stage-badge">
                            <span class="stage-dot"></span>
                            ${data.stage}: ${data.risk_level}
                        </div>
                    </div>

                    <!-- Metrics Row -->
                    <div class="metrics-row">
                        <div class="metric-box">
                            <span class="metric-label">Confidence</span>
                            <span class="metric-value">${data.confidence}%</span>
                            <div class="metric-sub">AI Certainty</div>
                        </div>
                        <div class="metric-box">
                            <span class="metric-label">Risk Level</span>
                            <span class="metric-value">${data.risk_level}</span>
                            <div class="metric-sub">Severity Assessment</div>
                        </div>
                    </div>

                    <!-- Clinical Summary -->
                    <div class="info-card">
                        <h3 class="card-title">
                            <svg style="width:20px;height:20px;" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                            Clinical Summary
                        </h3>
                        <p class="card-text">${data.summary}</p>
                    </div>

                    <!-- Recommendation -->
                    <div class="info-card rec-card">
                        <h3 class="card-title">
                            <svg style="width:20px;height:20px;" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"></path></svg>
                            Recommendation
                        </h3>
                        <p class="card-text">${data.recommendation}</p>
                    </div>

                    ${heatmapHtml}

                    <div class="card-footer">
                        Generated by DermaAI • Clinical Decision Support System
                    </div>
                </div>
            </div>
        `;

        resultDisplay.innerHTML = html;
        resultDisplay.classList.remove('hidden');
        resultDisplay.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function showError(msg) {
        errorBox.textContent = `⚠️ ${msg}`;
        errorBox.classList.remove('hidden');
        // Scroll to error if visible
        errorBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function clearError() {
        errorBox.classList.add('hidden');
        errorBox.textContent = '';
    }
});

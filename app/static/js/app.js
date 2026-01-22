/**
 * VoxDoc - Voice Symptom Intake & Documentation Assistant
 * Dark Theme Edition - JavaScript Controller
 */

// =====================================================
// DOM ELEMENTS
// =====================================================
const elements = {
    // Sidebar
    sidebar: document.getElementById('sidebar'),
    sidebarOverlay: document.getElementById('sidebarOverlay'),
    openSidebarBtn: document.getElementById('openSidebar'),
    closeSidebarBtn: document.getElementById('closeSidebar'),

    // Voice Recording
    recordBtn: document.getElementById('recordBtn'),
    recordRipple: document.getElementById('recordRipple'),
    micIcon: document.getElementById('micIcon'),
    stopIcon: document.getElementById('stopIcon'),
    waveformContainer: document.getElementById('waveformContainer'),
    voiceStatus: document.getElementById('voiceStatus'),
    durationDisplay: document.getElementById('durationDisplay'),
    recordingTime: document.getElementById('recordingTime'),

    // Text Input
    textInput: document.getElementById('textInput'),

    // Submit
    submitBtn: document.getElementById('submitBtn'),

    // Transcript Panel
    transcriptCard: document.getElementById('transcriptCard'),
    transcriptTitle: document.getElementById('transcriptTitle'),
    liveIndicator: document.getElementById('liveIndicator'),
    emptyState: document.getElementById('emptyState'),
    loadingState: document.getElementById('loadingState'),
    resultsContainer: document.getElementById('resultsContainer'),

    // Results
    transcriptionText: document.getElementById('transcriptionText'),
    chiefComplaint: document.getElementById('chiefComplaint'),
    symptomDetails: document.getElementById('symptomDetails'),
    soapNote: document.getElementById('soapNote'),
    audioPlaybackSection: document.getElementById('audioPlaybackSection'),
    resultAudioPlayer: document.getElementById('resultAudioPlayer'),
    textInputIndicator: document.getElementById('textInputIndicator'),

    // Actions
    copyBtn: document.getElementById('copyBtn'),
    exportBtn: document.getElementById('exportBtn')
};

// =====================================================
// STATE
// =====================================================
const state = {
    isRecording: false,
    mediaRecorder: null,
    audioChunks: [],
    audioBlob: null,
    audioUrl: null,
    recordingStartTime: null,
    recordingInterval: null,
    currentDocumentation: null
};

// =====================================================
// INITIALIZATION
// =====================================================
function init() {
    setupSidebar();
    setupRecording();
    setupTextInput();
    setupSubmit();
    setupActions();
}

// =====================================================
// SIDEBAR
// =====================================================
function setupSidebar() {
    elements.openSidebarBtn?.addEventListener('click', openSidebar);
    elements.closeSidebarBtn?.addEventListener('click', closeSidebar);
    elements.sidebarOverlay?.addEventListener('click', closeSidebar);

    // Nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
        });
    });
}

function openSidebar() {
    elements.sidebar?.classList.add('open');
    elements.sidebarOverlay?.classList.add('active');
}

function closeSidebar() {
    elements.sidebar?.classList.remove('open');
    elements.sidebarOverlay?.classList.remove('active');
}

// =====================================================
// RECORDING
// =====================================================
function setupRecording() {
    elements.recordBtn?.addEventListener('click', toggleRecording);
}

async function toggleRecording() {
    if (state.isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        state.mediaRecorder = new MediaRecorder(stream);
        state.audioChunks = [];

        state.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                state.audioChunks.push(e.data);
            }
        };

        state.mediaRecorder.onstop = () => {
            state.audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
            state.audioUrl = URL.createObjectURL(state.audioBlob);
            stream.getTracks().forEach(track => track.stop());
            updateSubmitButton();
        };

        state.mediaRecorder.start(100);
        state.isRecording = true;
        state.recordingStartTime = Date.now();

        // UI Updates
        elements.recordBtn?.classList.add('recording');
        elements.micIcon?.classList.add('hidden');
        elements.stopIcon?.classList.remove('hidden');
        elements.waveformContainer?.classList.add('recording');
        elements.recordRipple?.classList.add('active');
        elements.voiceStatus.textContent = 'Listening and transcribing...';
        elements.durationDisplay?.classList.remove('hidden');

        // Start timer
        state.recordingInterval = setInterval(updateRecordingTime, 1000);

    } catch (error) {
        console.error('Microphone access denied:', error);
        alert('Microphone access is required for voice recording.');
    }
}

function stopRecording() {
    if (state.mediaRecorder && state.isRecording) {
        state.mediaRecorder.stop();
        state.isRecording = false;

        // UI Updates
        elements.recordBtn?.classList.remove('recording');
        elements.micIcon?.classList.remove('hidden');
        elements.stopIcon?.classList.add('hidden');
        elements.waveformContainer?.classList.remove('recording');
        elements.recordRipple?.classList.remove('active');
        elements.voiceStatus.textContent = 'Recording complete. Ready to generate documentation.';

        // Stop timer
        clearInterval(state.recordingInterval);
    }
}

function updateRecordingTime() {
    if (!state.recordingStartTime) return;

    const elapsed = Math.floor((Date.now() - state.recordingStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const seconds = (elapsed % 60).toString().padStart(2, '0');

    if (elements.recordingTime) {
        elements.recordingTime.textContent = `${minutes}:${seconds}`;
    }
}

// =====================================================
// TEXT INPUT
// =====================================================
function setupTextInput() {
    elements.textInput?.addEventListener('input', () => {
        updateSubmitButton();
    });
}

function updateSubmitButton() {
    const hasAudio = state.audioBlob !== null;
    const hasText = elements.textInput?.value.trim().length > 0;

    if (elements.submitBtn) {
        elements.submitBtn.disabled = !(hasAudio || hasText);
    }
}

// =====================================================
// SUBMIT & PROCESSING
// =====================================================
function setupSubmit() {
    elements.submitBtn?.addEventListener('click', processInput);
}

async function processInput() {
    const hasAudio = state.audioBlob !== null;
    const textContent = elements.textInput?.value.trim();

    if (!hasAudio && !textContent) return;

    showLoading();

    try {
        let response;

        if (hasAudio) {
            // Voice input: Use /api/voice-intake with FormData
            const formData = new FormData();
            formData.append('audio', state.audioBlob, 'recording.webm');

            response = await fetch('/api/voice-intake', {
                method: 'POST',
                body: formData
            });
        } else {
            // Text input: Use /api/document with JSON
            response = await fetch('/api/document', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ transcript: textContent })
            });
        }

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Processing failed');
        }

        const data = await response.json();

        // Normalize response format (text endpoint doesn't include transcript)
        if (!hasAudio) {
            data.transcript = textContent;
            data.duration_seconds = 0;
        }

        state.currentDocumentation = data;
        displayResults(data);

    } catch (error) {
        console.error('Processing error:', error);
        showError(error.message);
    }
}

// =====================================================
// UI STATE MANAGEMENT
// =====================================================
function showLoading() {
    elements.emptyState?.classList.add('hidden');
    elements.resultsContainer?.classList.add('hidden');
    elements.loadingState?.classList.remove('hidden');
    elements.transcriptTitle.textContent = 'Processing...';
}

function showResults() {
    elements.emptyState?.classList.add('hidden');
    elements.loadingState?.classList.add('hidden');
    elements.resultsContainer?.classList.remove('hidden');
    elements.transcriptTitle.textContent = 'Documentation Results';
}

function showError(message) {
    elements.loadingState?.classList.add('hidden');
    elements.transcriptTitle.textContent = 'Error';

    // Create error display
    const errorHtml = `
        <div style="padding: 20px; background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; color: #dc2626;">
            <strong>Error:</strong> ${message}
        </div>
    `;

    if (elements.resultsContainer) {
        elements.resultsContainer.innerHTML = errorHtml;
        elements.resultsContainer.classList.remove('hidden');
    }
}

function displayResults(data) {
    showResults();

    // Transcription
    if (elements.transcriptionText) {
        elements.transcriptionText.textContent = data.transcript;
    }

    // Audio Playback (only for voice input)
    if (state.audioUrl && elements.audioPlaybackSection && elements.resultAudioPlayer) {
        elements.resultAudioPlayer.src = state.audioUrl;
        elements.audioPlaybackSection.classList.remove('hidden');
        elements.textInputIndicator?.classList.add('hidden');
    } else {
        elements.audioPlaybackSection?.classList.add('hidden');
        elements.textInputIndicator?.classList.remove('hidden');
    }

    // Documentation fields
    const doc = data.documentation;

    if (elements.chiefComplaint) {
        elements.chiefComplaint.textContent = doc.chief_complaint || 'N/A';
    }

    // Symptom Details - Map backend field names to display
    if (elements.symptomDetails && doc.symptom_details) {
        const details = doc.symptom_details;
        // symptoms_mentioned is an array, join for display
        const symptomsText = Array.isArray(details.symptoms_mentioned)
            ? details.symptoms_mentioned.join(', ')
            : (details.symptoms_mentioned || 'not specified');

        elements.symptomDetails.innerHTML = `
            <ul>
                <li><strong>Symptoms:</strong> ${symptomsText}</li>
                <li><strong>Onset:</strong> ${details.onset || 'not specified'}</li>
                <li><strong>Duration:</strong> ${details.duration || 'not specified'}</li>
                <li><strong>Location:</strong> ${details.location || 'not specified'}</li>
                <li><strong>Aggravating Factors:</strong> ${details.aggravating_factors || 'not specified'}</li>
                <li><strong>Severity:</strong> ${details.severity_description || 'not specified'}</li>
            </ul>
        `;
    }

    // SOAP Note - Backend uses soap_note_subjective
    if (elements.soapNote) {
        elements.soapNote.textContent = doc.soap_note_subjective || 'Patient describes symptoms.';
    }
}

// =====================================================
// ACTIONS (Copy, Export)
// =====================================================
function setupActions() {
    elements.copyBtn?.addEventListener('click', copyToClipboard);
    elements.exportBtn?.addEventListener('click', exportJSON);
}

async function copyToClipboard() {
    if (!state.currentDocumentation) return;

    const doc = state.currentDocumentation.documentation;
    const text = `
CHIEF COMPLAINT: ${doc.chief_complaint}

SYMPTOM DETAILS:
- Symptoms: ${doc.symptom_details?.symptoms || 'N/A'}
- Onset: ${doc.symptom_details?.onset || 'N/A'}
- Duration: ${doc.symptom_details?.duration || 'N/A'}
- Location: ${doc.symptom_details?.location || 'N/A'}

SOAP NOTE (Subjective):
${doc.soap_subjective}
    `.trim();

    try {
        await navigator.clipboard.writeText(text);

        // Visual feedback
        const originalSvg = elements.copyBtn.innerHTML;
        elements.copyBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12" />
            </svg>
        `;
        elements.copyBtn.style.color = '#10b981';

        setTimeout(() => {
            elements.copyBtn.innerHTML = originalSvg;
            elements.copyBtn.style.color = '';
        }, 2000);

    } catch (error) {
        console.error('Copy failed:', error);
    }
}

function exportJSON() {
    if (!state.currentDocumentation) return;

    const dataStr = JSON.stringify(state.currentDocumentation, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `voxdoc_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// =====================================================
// INITIALIZE
// =====================================================
document.addEventListener('DOMContentLoaded', init);

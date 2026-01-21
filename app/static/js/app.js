/**
 * Voice Symptom Intake & Documentation Assistant - Frontend Logic
 * 
 * Handles audio recording, file upload, API communication, and results display.
 */

// State management
const state = {
    mediaRecorder: null,
    audioChunks: [],
    audioBlob: null,
    audioUrl: null,
    isRecording: false,
    textInput: ''  // Text input state
};

// DOM elements
const elements = {
    recordBtn: document.getElementById('recordBtn'),
    stopBtn: document.getElementById('stopBtn'),
    playBtn: document.getElementById('playBtn'),
    submitBtn: document.getElementById('submitBtn'),
    textInput: document.getElementById('textInput'),  // Text input element
    resultsSection: document.getElementById('resultsSection'),
    loadingIndicator: document.getElementById('loadingIndicator'),
    transcriptionCard: document.getElementById('transcriptionCard'),
    documentationCard: document.getElementById('documentationCard'),
    errorCard: document.getElementById('errorCard'),
    transcriptionText: document.getElementById('transcriptionText'),
    audioDuration: document.getElementById('audioDuration'),
    chiefComplaint: document.getElementById('chiefComplaint'),
    symptomDetails: document.getElementById('symptomDetails'),
    soapNote: document.getElementById('soapNote'),
    errorText: document.getElementById('errorText'),
    exportJsonBtn: document.getElementById('exportJsonBtn'),
    copyBtn: document.getElementById('copyBtn'),
    visualizerCanvas: document.getElementById('visualizerCanvas'),
    audioPlaybackSection: document.getElementById('audioPlaybackSection'),
    resultAudioPlayer: document.getElementById('resultAudioPlayer')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkApiHealth();
});

/**
 * Initialize all event listeners
 */
function initializeEventListeners() {
    elements.recordBtn.addEventListener('click', startRecording);
    elements.stopBtn.addEventListener('click', stopRecording);
    elements.playBtn.addEventListener('click', playRecording);
    elements.submitBtn.addEventListener('click', submitForDocumentation);
    elements.exportJsonBtn.addEventListener('click', exportAsJson);
    elements.copyBtn.addEventListener('click', copyToClipboard);

    // Text input listener
    elements.textInput.addEventListener('input', handleTextInput);
}

/**
 * Check API health status
 */
async function checkApiHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('API Health:', data);
    } catch (error) {
        console.error('API health check failed:', error);
        showError('Unable to connect to backend services. Please ensure the server is running.');
    }
}

/**
 * Start audio recording
 */
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        state.mediaRecorder = new MediaRecorder(stream);
        state.audioChunks = [];

        state.mediaRecorder.ondataavailable = (event) => {
            state.audioChunks.push(event.data);
        };

        state.mediaRecorder.onstop = async () => {
            // Create a blob from recorded chunks (this is WebM/Opus format)
            const webmBlob = new Blob(state.audioChunks, { type: 'audio/webm' });

            // Convert to WAV using Web Audio API
            const arrayBuffer = await webmBlob.arrayBuffer();
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

            // Convert AudioBuffer to WAV
            state.audioBlob = audioBufferToWav(audioBuffer);
            state.audioUrl = URL.createObjectURL(state.audioBlob);

            // Enable playback and submit buttons
            elements.playBtn.disabled = false;
            elements.submitBtn.disabled = false;

            // Stop all audio tracks
            stream.getTracks().forEach(track => track.stop());
        };

        state.mediaRecorder.start();
        state.isRecording = true;

        // Update UI
        elements.recordBtn.disabled = true;
        elements.stopBtn.disabled = false;
        elements.recordBtn.textContent = '🔴 Recording...';

        // Start visualizer
        visualizeAudio(stream);

    } catch (error) {
        console.error('Error starting recording:', error);
        showError('Unable to access microphone. Please check permissions.');
    }
}

/**
 * Stop audio recording
 */
function stopRecording() {
    if (state.mediaRecorder && state.isRecording) {
        state.mediaRecorder.stop();
        state.isRecording = false;

        // Update UI
        elements.recordBtn.disabled = false;
        elements.stopBtn.disabled = true;
        elements.recordBtn.innerHTML = '<span class="btn-icon">🎤</span> Start Recording';
    }
}

/**
 * Play recorded audio
 */
function playRecording() {
    if (state.audioUrl) {
        const audio = new Audio(state.audioUrl);
        audio.play();
    }
}

/**
 * Handle text input
 */
function handleTextInput(event) {
    const text = event.target.value.trim();
    state.textInput = text;

    // Enable submit if there's text
    if (text.length > 0) {
        elements.submitBtn.disabled = false;
        // Clear audio state when typing
        state.audioBlob = null;
    } else if (!state.audioBlob) {
        elements.submitBtn.disabled = true;
    }
}

/**
 * Submit audio or text for documentation
 */
async function submitForDocumentation() {
    // Determine input source: text first, then audio
    const textToSubmit = state.textInput;
    const audioToSubmit = state.audioBlob;

    if (!textToSubmit && !audioToSubmit) {
        showError('Please record audio or type symptoms.');
        return;
    }

    // Show loading, hide previous results
    elements.resultsSection.style.display = 'block';
    elements.loadingIndicator.style.display = 'block';
    elements.transcriptionCard.style.display = 'none';
    elements.documentationCard.style.display = 'none';
    elements.errorCard.style.display = 'none';

    try {
        let data;

        if (textToSubmit) {
            // Text input mode - call document endpoint directly
            const response = await fetch('/api/document', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ transcript: textToSubmit })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Request failed');
            }

            const docData = await response.json();

            // Format as voice-intake style response
            data = {
                transcript: textToSubmit,
                documentation: docData.documentation,
                duration_seconds: 0,  // No audio duration for text
                requires_clinician_review: docData.requires_clinician_review,
                compliance_notice: docData.compliance_notice
            };
        } else {
            // Audio mode - call voice-intake endpoint
            const formData = new FormData();
            formData.append('audio', audioToSubmit, 'recording.wav');

            const response = await fetch('/api/voice-intake', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Request failed');
            }

            data = await response.json();
        }

        // Hide loading
        elements.loadingIndicator.style.display = 'none';

        // Display results
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        elements.loadingIndicator.style.display = 'none';
        showError(error.message || 'Failed to process input. Please try again.');
    }
}

/**
 * Display documentation results
 */
function displayResults(data) {
    // Transcription
    elements.transcriptionText.textContent = data.transcript;
    elements.audioDuration.textContent = data.duration_seconds.toFixed(1);
    elements.transcriptionCard.style.display = 'block';

    // Audio Playback for Verification (only show if recorded audio was used)
    if (state.audioUrl) {
        elements.resultAudioPlayer.src = state.audioUrl;
        elements.audioPlaybackSection.style.display = 'block';
    } else {
        // Hide audio section for text input
        elements.audioPlaybackSection.style.display = 'none';
    }

    // Documentation
    const doc = data.documentation;

    // Chief Complaint
    elements.chiefComplaint.textContent = doc.chief_complaint || 'N/A';

    // Symptom Details
    if (doc.symptom_details) {
        const details = doc.symptom_details;
        let html = '<ul>';

        if (details.symptoms_mentioned) {
            html += '<li><strong>Symptoms:</strong> ' + details.symptoms_mentioned.join(', ') + '</li>';
        }
        if (details.onset) {
            html += '<li><strong>Onset:</strong> ' + details.onset + '</li>';
        }
        if (details.duration) {
            html += '<li><strong>Duration:</strong> ' + details.duration + '</li>';
        }
        if (details.location) {
            html += '<li><strong>Location:</strong> ' + details.location + '</li>';
        }
        if (details.quality) {
            html += '<li><strong>Quality:</strong> ' + details.quality + '</li>';
        }
        if (details.severity_description) {
            html += '<li><strong>Severity (patient description):</strong> ' + details.severity_description + '</li>';
        }
        if (details.associated_symptoms && details.associated_symptoms.length > 0) {
            html += '<li><strong>Associated Symptoms:</strong> ' + details.associated_symptoms.join(', ') + '</li>';
        }

        html += '</ul>';
        elements.symptomDetails.innerHTML = html;
    }

    // SOAP Note
    elements.soapNote.textContent = doc.soap_note_subjective || 'N/A';

    // Store data for export
    window.currentDocumentation = data;

    elements.documentationCard.style.display = 'block';
}

/**
 * Show error message
 */
function showError(message) {
    elements.errorText.textContent = message;
    elements.errorCard.style.display = 'block';
    elements.resultsSection.style.display = 'block';
}

/**
 * Export documentation as JSON
 */
function exportAsJson() {
    if (!window.currentDocumentation) {
        alert('No documentation to export');
        return;
    }

    const dataStr = JSON.stringify(window.currentDocumentation, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement('a');
    link.href = url;
    link.download = 'symptom_documentation_' + new Date().getTime() + '.json';
    link.click();

    URL.revokeObjectURL(url);
}

/**
 * Copy documentation to clipboard
 */
async function copyToClipboard() {
    if (!window.currentDocumentation) {
        alert('No documentation to copy');
        return;
    }

    const doc = window.currentDocumentation.documentation;

    let text = '=== VOICE SYMPTOM INTAKE DOCUMENTATION ===\n\n';
    text += 'COMPLIANCE NOTICE: Administrative documentation only. Requires clinician review.\n\n';
    text += 'CHIEF COMPLAINT:\n' + (doc.chief_complaint || 'N/A') + '\n\n';
    text += 'SOAP NOTE - SUBJECTIVE:\n' + (doc.soap_note_subjective || 'N/A') + '\n\n';
    text += 'TRANSCRIPTION:\n' + window.currentDocumentation.transcript + '\n';

    try {
        await navigator.clipboard.writeText(text);
        alert('Documentation copied to clipboard!');
    } catch (error) {
        console.error('Failed to copy:', error);
        alert('Failed to copy to clipboard');
    }
}

/**
 * Visualize audio waveform
 */
function visualizeAudio(stream) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    const microphone = audioContext.createMediaStreamSource(stream);

    microphone.connect(analyser);
    analyser.fftSize = 256;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const canvas = elements.visualizerCanvas;
    const canvasCtx = canvas.getContext('2d');

    function draw() {
        if (!state.isRecording) return;

        requestAnimationFrame(draw);

        analyser.getByteTimeDomainData(dataArray);

        canvasCtx.fillStyle = 'rgb(244, 242, 255)';  // Direct Care lavender-soft
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = 'rgb(138, 99, 210)';  // Direct Care brand purple
        canvasCtx.beginPath();

        const sliceWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;

            if (i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        canvasCtx.lineTo(canvas.width, canvas.height / 2);
        canvasCtx.stroke();
    }

    draw();
}

/**
 * Convert AudioBuffer to WAV blob
 */
function audioBufferToWav(audioBuffer) {
    const numChannels = 1; // Force mono
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    // Get audio data from first channel (or mix down to mono)
    let audioData;
    if (audioBuffer.numberOfChannels === 1) {
        audioData = audioBuffer.getChannelData(0);
    } else {
        // Mix down to mono by averaging channels
        const left = audioBuffer.getChannelData(0);
        const right = audioBuffer.getChannelData(1);
        audioData = new Float32Array(left.length);
        for (let i = 0; i < left.length; i++) {
            audioData[i] = (left[i] + right[i]) / 2;
        }
    }

    const dataLength = audioData.length * (bitDepth / 8);
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    // Write WAV header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * (bitDepth / 8), true); // byte rate
    view.setUint16(32, numChannels * (bitDepth / 8), true); // block align
    view.setUint16(34, bitDepth, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataLength, true);

    // Write audio data
    floatTo16BitPCM(view, 44, audioData);

    return new Blob([view], { type: 'audio/wav' });
}

/**
 * Helper function to write string to DataView
 */
function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

/**
 * Helper function to convert float32 to 16-bit PCM
 */
function floatTo16BitPCM(view, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, input[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

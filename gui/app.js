const API_URL = "http://127.0.0.1:8000";

let activePersona = 0;
let isDrawing = false;
let canvasHasDrawing = false;

// DOM Elements
const chatFeed = document.getElementById("chat-feed");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const personaList = document.getElementById("persona-list");
const paintCanvas = document.getElementById("paint-canvas");
const clearCanvasBtn = document.getElementById("clear-canvas");
const brushColorInput = document.getElementById("brush-color");
const memoryGrid = document.getElementById("memory-grid");

// Audio Sliders
const audioPitch = document.getElementById("audio-pitch");
const audioVol = document.getElementById("audio-vol");
const audioCentroid = document.getElementById("audio-centroid");
const audioZcr = document.getElementById("audio-zcr");
const sendAudioToggle = document.getElementById("send-audio-toggle");

const pitchVal = document.getElementById("pitch-val");
const volVal = document.getElementById("vol-val");
const centroidVal = document.getElementById("centroid-val");
const zcrVal = document.getElementById("zcr-val");

// Update Audio slider value displays
function setupAudioSliders() {
    const updateLabels = () => {
        pitchVal.innerText = parseFloat(audioPitch.value).toFixed(1);
        volVal.innerText = parseFloat(audioVol.value).toFixed(1);
        centroidVal.innerText = parseFloat(audioCentroid.value).toFixed(1);
        zcrVal.innerText = parseFloat(audioZcr.value).toFixed(1);
    };
    [audioPitch, audioVol, audioCentroid, audioZcr].forEach(slider => {
        slider.addEventListener("input", updateLabels);
    });
    updateLabels();
}

// Initialize Canvas
const ctx = paintCanvas.getContext("2d");
ctx.fillStyle = "#000000";
ctx.fillRect(0, 0, paintCanvas.width, paintCanvas.height);
ctx.strokeStyle = brushColorInput.value;
ctx.lineWidth = 6;
ctx.lineCap = "round";

// Canvas Event Listeners
paintCanvas.addEventListener("mousedown", (e) => {
    isDrawing = true;
    draw(e);
});

paintCanvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", () => isDrawing = false);

function draw(e) {
    if (!isDrawing) return;
    const rect = paintCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    ctx.strokeStyle = brushColorInput.value;
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
    canvasHasDrawing = true;
}

clearCanvasBtn.addEventListener("click", () => {
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, paintCanvas.width, paintCanvas.height);
    ctx.beginPath();
    canvasHasDrawing = false;
});

// Load Personas
async function loadPersonas() {
    try {
        const res = await fetch(`${API_URL}/personas`);
        const personas = await res.json();
        
        personaList.innerHTML = "";
        personas.forEach((p, idx) => {
            const card = document.createElement("div");
            card.className = `persona-card ${idx === activePersona ? "active" : ""}`;
            card.innerHTML = `
                <div class="persona-name">${p.name}</div>
                <div class="persona-desc">${p.description}</div>
            `;
            card.addEventListener("click", () => {
                document.querySelectorAll(".persona-card").forEach(c => c.classList.remove("active"));
                card.classList.add("active");
                activePersona = p.id;
            });
            personaList.appendChild(card);
        });
    } catch (err) {
        console.error("Failed to load personas:", err);
    }
}

// Render Memory Grid Slots
function initMemoryGrid() {
    memoryGrid.innerHTML = "";
    // 4 slots * 16 visual dimensions = 64 cells
    for (let slot = 0; slot < 4; slot++) {
        for (let dim = 0; dim < 16; dim++) {
            const cell = document.createElement("div");
            cell.className = "mem-cell";
            cell.id = `mem-cell-${slot}-${dim}`;
            cell.innerHTML = `
                <span class="mem-cell-val">0.00</span>
                <span style="font-size:7px; opacity:0.5">S${slot} D${dim}</span>
            `;
            memoryGrid.appendChild(cell);
        }
    }
}

function updateMemoryGrid(memoryData) {
    if (!memoryData) return;
    memoryData.forEach((slotData, slotIdx) => {
        slotData.forEach((val, dimIdx) => {
            const cell = document.getElementById(`mem-cell-${slotIdx}-${dimIdx}`);
            if (cell) {
                const num = val.toFixed(2);
                cell.querySelector(".mem-cell-val").innerText = num;
                
                // Color intensity based on value (-1 to 1)
                const normVal = Math.min(Math.max((val + 1) / 2, 0), 1);
                cell.style.backgroundColor = `rgba(0, 255, 204, ${normVal * 0.4})`;
                cell.style.borderColor = `rgba(0, 255, 204, ${normVal * 0.6})`;
            }
        });
    });
}

// Update Emotion Meters
function updateEmotions(emotions) {
    if (!emotions) return;
    const scale = (val) => Math.round(((val + 1) / 2) * 100);
    for (const [key, val] of Object.entries(emotions)) {
        const lowerKey = key.toLowerCase();
        const fillEl = document.getElementById(`bar-${lowerKey}`);
        const textEl = document.getElementById(`val-${lowerKey}`);
        
        if (fillEl && textEl) {
            const pct = scale(val);
            fillEl.style.width = `${pct}%`;
            textEl.innerText = val.toFixed(2);
        }
    }
}

// Append Message Helper
function appendMessage(sender, text, imgBase64 = null, hasAudio = false) {
    const msg = document.createElement("div");
    msg.className = `message ${sender}`;
    
    if (imgBase64) {
        const img = document.createElement("img");
        img.src = imgBase64;
        img.className = "msg-attachment";
        msg.appendChild(img);
    }
    
    const textNode = document.createElement("span");
    let bubbleText = text;
    if (hasAudio) {
        bubbleText = "🎵 [Continuous Audio Waves Attached] " + bubbleText;
    }
    textNode.innerText = bubbleText;
    msg.appendChild(textNode);
    
    chatFeed.appendChild(msg);
    chatFeed.scrollTop = chatFeed.scrollHeight;
}

// Send Message Flow
async function sendMessage() {
    const text = chatInput.value.trim();
    const isAudioChecked = sendAudioToggle.checked;
    if (!text && !canvasHasDrawing && !isAudioChecked) return;
    
    let imgBase64 = null;
    if (canvasHasDrawing) {
        imgBase64 = paintCanvas.toDataURL("image/png");
    }
    
    let audioFeatures = null;
    if (isAudioChecked) {
        // Construct continuous audio sequence (2 time steps representing current sliders)
        const p = parseFloat(audioPitch.value);
        const v = parseFloat(audioVol.value);
        const c = parseFloat(audioCentroid.value);
        const z = parseFloat(audioZcr.value);
        audioFeatures = [
            [p, v, c, z],
            [p * 0.9, v * 0.95, c * 1.05, z * 0.9] // Next step features
        ];
    }
    
    // UI Feedback
    appendMessage("user", text || "[Visual/Audio Input]", imgBase64, isAudioChecked);
    chatInput.value = "";
    
    try {
        const res = await fetch(`${API_URL}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: text,
                persona: activePersona,
                image_base64: imgBase64,
                audio_features: audioFeatures
            })
        });
        
        if (!res.ok) {
            throw new Error("Server error running upgraded inference.");
        }
        
        const data = await res.json();
        
        // Append response
        appendMessage("assistant", data.response);
        
        // Update stats
        updateEmotions(data.emotions);
        updateMemoryGrid(data.memory);
        
    } catch (err) {
        console.error(err);
        appendMessage("system", "Error communicating with PyAgu backend. Is the server running?");
    }
}

// Listeners
sendBtn.addEventListener("click", sendMessage);
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
});

// Boot
loadPersonas();
initMemoryGrid();
setupAudioSliders();

// Auto reconnect
setInterval(() => {
    if (document.querySelectorAll(".persona-card").length === 0) {
        loadPersonas();
    }
}, 3000);

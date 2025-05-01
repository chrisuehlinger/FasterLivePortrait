import * as THREE from 'three';

// Configuration
const SERVER_URL = window.location.hostname + ":" + window.location.port;
const USE_SSL = window.location.protocol === 'https:';
const WS_URL = `${USE_SSL ? 'wss' : 'ws'}://${SERVER_URL}`;

// Elements
const sessionInputDiv = document.getElementById('sessionInput');
const sessionIdInput = document.getElementById('sessionIdInput');
const connectButton = document.getElementById('connectButton');
const connectionStatus = document.getElementById('connectionStatus');

// Global variables
let websocket;
let sessionId = null;
let frameCount = 0;
let lastFrameTime = 0;
let fps = 0;
let statusFadeTimeout = null;
let isConnected = false;

// Three.js variables
let scene, camera, renderer;
let videoTexture, videoMaterial, videoMesh;
let rendererWidth = window.innerWidth;
let rendererHeight = window.innerHeight;

// Initialize Three.js scene
function initThreeJS() {
    // Create scene
    scene = new THREE.Scene();
    
    // Create camera (orthographic to prevent distortion)
    camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    
    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    
    // Create a placeholder texture until we receive video frames
    videoTexture = new THREE.Texture();
    videoTexture.minFilter = THREE.LinearFilter;
    videoTexture.magFilter = THREE.LinearFilter;
    
    // Create material using the texture
    videoMaterial = new THREE.MeshBasicMaterial({ 
        map: videoTexture,
        side: THREE.FrontSide
    });
    
    // Create a plane geometry that fills the screen
    const planeGeometry = new THREE.PlaneGeometry(2, 2);
    
    // Create mesh with the geometry and material
    videoMesh = new THREE.Mesh(planeGeometry, videoMaterial);
    
    // Add mesh to scene
    scene.add(videoMesh);
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

// Handle window resize
function onWindowResize() {
    rendererWidth = window.innerWidth;
    rendererHeight = window.innerHeight;
    
    renderer.setSize(rendererWidth, rendererHeight);
    
    // Update video texture aspect ratio if we have frame dimensions
    if (videoTexture.image) {
        updateTextureAspectRatio();
    }
}

// Update texture aspect ratio to maintain proper dimensions
function updateTextureAspectRatio() {
    const videoAspect = videoTexture.image.width / videoTexture.image.height;
    const screenAspect = rendererWidth / rendererHeight;
    
    // Scale the plane to maintain aspect ratio
    if (videoAspect > screenAspect) {
        // Video is wider than screen
        videoMesh.scale.set(1, screenAspect / videoAspect, 1);
    } else {
        // Video is taller than screen
        videoMesh.scale.set(videoAspect / screenAspect, 1, 1);
    }
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

// Connect to WebSocket
function connectToWebSocket() {
    if (websocket) {
        websocket.close();
    }
    
    if (!sessionId) {
        showStatus('Error: No session ID provided. Cannot connect to stream.', true);
        return;
    }
    
    websocket = new WebSocket(`${WS_URL}/ws/viewer/${sessionId}`);
    
    websocket.onopen = () => {
        isConnected = true;
        showStatus('Connected to stream. Waiting for video...', false);
        
        // Send heartbeat every 30 seconds to keep connection alive
        setInterval(() => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send('heartbeat');
            }
        }, 30000);
    };
    
    websocket.onclose = (event) => {
        isConnected = false;
        if (event.wasClean) {
            showStatus(`Connection closed: ${event.reason}`, true);
        } else {
            showStatus('Connection lost. Attempting to reconnect...', true);
            setTimeout(connectToWebSocket, 3000);
        }
    };
    
    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        showStatus('Connection error. Please try again later.', true);
    };
    
    websocket.onmessage = handleWebSocketMessage;
}

// Handle incoming WebSocket messages
async function handleWebSocketMessage(event) {
    if (typeof event.data === 'string') {
        // Handle text messages (like status updates)
        console.log('Received text message:', event.data);
        return;
    }
    
    try {
        // Handle binary data (frames)
        const blob = event.data;
        
        // Create an image from the blob
        const imageBitmap = await createImageBitmap(blob);
        
        // Update the texture with new frame
        if (videoTexture.image !== imageBitmap) {
            // Clean up previous texture if it exists
            if (videoTexture.image) {
                videoTexture.dispose();
            }
            
            // Create new texture with the received frame
            videoTexture.image = imageBitmap;
            videoTexture.needsUpdate = true;
            
            // Update texture aspect ratio
            updateTextureAspectRatio();
        }
        
        // Hide connection UI if it's still visible
        if (sessionInputDiv.style.display !== 'none') {
            sessionInputDiv.style.display = 'none';
        }
        
        // Update FPS counter
        frameCount++;
        const now = performance.now();
        const elapsed = now - lastFrameTime;
        
        if (elapsed >= 1000) { // Update FPS every second
            fps = Math.round((frameCount * 1000) / elapsed);
            frameCount = 0;
            lastFrameTime = now;
            
            // Update status with FPS
            showStatus(`FPS: ${fps}`, false, true);
        }
    } catch (error) {
        console.error('Error processing video frame:', error);
        showStatus('Error processing video frame', true);
    }
}

// Show status message with optional fade
function showStatus(message, isError, autoFade = false) {
    connectionStatus.textContent = message;
    connectionStatus.style.color = isError ? 'rgba(255, 80, 80, 0.8)' : 'rgba(255, 255, 255, 0.7)';
    connectionStatus.classList.remove('fade');
    
    // Clear any existing timeout
    if (statusFadeTimeout) {
        clearTimeout(statusFadeTimeout);
    }
    
    // Auto fade the status message after a delay if requested
    if (autoFade && isConnected) {
        statusFadeTimeout = setTimeout(() => {
            connectionStatus.classList.add('fade');
        }, 3000);
    }
}

// Get session ID from URL
function getSessionIdFromUrl() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('session');
}

// Event listeners
connectButton.addEventListener('click', () => {
    sessionId = sessionIdInput.value.trim();
    if (!sessionId) {
        showStatus('Please enter a session ID', true);
        return;
    }
    
    connectToWebSocket();
});

// Initialize when page loads
function initialize() {
    // Initialize Three.js
    initThreeJS();
    
    // Start animation loop
    animate();
    
    // Check for session ID in URL
    const urlSessionId = getSessionIdFromUrl();
    if (urlSessionId) {
        sessionId = urlSessionId;
        sessionIdInput.value = sessionId;
        connectToWebSocket();
    }
    
    // Start FPS timer
    lastFrameTime = performance.now();
    
    // Setup keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // F or f key to toggle fullscreen
        if (e.key === 'f' || e.key === 'F') {
            toggleFullscreen();
        }
    });
}

// Toggle fullscreen
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(err => {
            console.error(`Error attempting to enable fullscreen: ${err.message}`);
        });
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        }
    }
}

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (websocket) {
        websocket.close();
    }
});

// Initialize the application
initialize();
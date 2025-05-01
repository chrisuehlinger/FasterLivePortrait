import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';

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
let backgroundTexture, backgroundMaterial, backgroundMesh;
let composer;
let rendererWidth = window.innerWidth;
let rendererHeight = window.innerHeight;

// Background images corresponding to each source
const backgroundImages = [
    '/frontend/images/ahau-kin-bg.png',  // Placeholder paths - update these later
    '/frontend/images/ix-chel-bg.png',
    '/frontend/images/chac-bolay-bg.png'
];
const keyColors = [
    new THREE.Color('yellow'),  // Placeholder paths - update these later
    new THREE.Color('blue'),
    new THREE.Color('red')
];
let currentBackgroundIndex = 0;

// Luma key shader for transparency
const lumaKeyShader = {
    uniforms: {
        'tDiffuse': { value: null },
        'threshold': { value: 0.05 }  // 5% luminosity threshold
    },
    vertexShader: `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    `,
    fragmentShader: `
        uniform sampler2D tDiffuse;
        uniform float threshold;
        varying vec2 vUv;
        
        void main() {
            vec4 color = texture2D(tDiffuse, vUv);
            float luminance = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
            
            if (luminance < threshold) {
                gl_FragColor = vec4(color.rgb, 0.0); // Transparent
            } else {
                gl_FragColor = color;
            }
        }
    `
};

// Initialize Three.js scene
function initThreeJS() {
    // Create scene
    scene = new THREE.Scene();
    
    // Create camera (orthographic to prevent distortion)
    camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    
    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    
    // Create background layer first (rendered behind)
    backgroundTexture = new THREE.TextureLoader().load(backgroundImages[0], updateTextureAspectRatio);
    backgroundTexture.minFilter = THREE.LinearFilter;
    backgroundTexture.magFilter = THREE.LinearFilter;
    
    // Create inverted background material using shader material
    backgroundMaterial = new THREE.ShaderMaterial({
        uniforms: {
            'backgroundTexture': { value: backgroundTexture },
            'threshold': { value: 0.05 },
            'keyColor': { value: keyColors[currentBackgroundIndex] }
        },
        vertexShader: `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform sampler2D backgroundTexture;
            uniform float threshold;
            uniform vec3 keyColor;
            varying vec2 vUv;
            void main() {
                vec4 texColor = texture2D(backgroundTexture, vUv);
                // Invert the colors
                vec3 inverted = 1.0 - texColor.rgb;

                float luminance = 0.299 * inverted.r + 0.587 * inverted.g + 0.114 * inverted.b;
                luminance = luminance * texColor.a;
                if (luminance < threshold) {
                    gl_FragColor = vec4(inverted.rgb, 0.0); // Transparent
                } else {
                    gl_FragColor = vec4(keyColor*luminance, texColor.a);
                }
            }
        `,
        side: THREE.FrontSide
    });
    
    const backgroundGeometry = new THREE.PlaneGeometry(2, 2);
    backgroundMesh = new THREE.Mesh(backgroundGeometry, backgroundMaterial);
    backgroundMesh.position.z = -0.1; // Place behind video layer
    scene.add(backgroundMesh);
    
    // Create a placeholder texture until we receive video frames
    videoTexture = new THREE.Texture();
    videoTexture.minFilter = THREE.LinearFilter;
    videoTexture.magFilter = THREE.LinearFilter;
    
    // Create material using the texture with transparency and flipping the video
    videoMaterial = new THREE.ShaderMaterial({
        uniforms: {
            'videoTexture': { value: videoTexture },
            'threshold': { value: 0.25 },
            'keyColor': { value: keyColors[currentBackgroundIndex] }
        },
        vertexShader: `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform sampler2D videoTexture;
            uniform float threshold;
            uniform vec3 keyColor;
            varying vec2 vUv;
            
            void main() {
                // Flip the Y coordinate to fix upside-down video
                vec2 flippedUv = vec2(vUv.x, 1.0 - vUv.y);
                vec4 color = texture2D(videoTexture, flippedUv);
                // Invert the colors
                vec3 inverted = 1.0 - color.rgb;

                float luminance = 0.299 * inverted.r + 0.587 * inverted.g + 0.114 * inverted.b;
                
                if (luminance < threshold) {
                    gl_FragColor = vec4(inverted.rgb, 0.0); // Transparent
                } else {
                    gl_FragColor = vec4(keyColor*luminance, color.a);
                }
            }
        `,
        transparent: true,
        side: THREE.FrontSide
    });
    
    // Create a plane geometry that fills the screen
    const planeGeometry = new THREE.PlaneGeometry(2, 2);
    
    // Create mesh with the geometry and material
    videoMesh = new THREE.Mesh(planeGeometry, videoMaterial);
    
    // Add mesh to scene
    scene.add(videoMesh);
    
    // Set up post-processing
    setupPostProcessing();
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

// Set up post-processing pipeline
function setupPostProcessing() {
    // Create composer
    composer = new EffectComposer(renderer);
    
    // Add render pass
    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);
    
    // Add luma key shader pass for the video
    // const lumaKeyPass = new ShaderPass(lumaKeyShader);
    // composer.addPass(lumaKeyPass);
    
    // You can add more post-processing effects here as needed
}

// Switch to a different background image
function switchBackground(index) {
    if (index >= 0 && index < backgroundImages.length) {
        currentBackgroundIndex = index;
        new THREE.TextureLoader().load(backgroundImages[index], (texture) => {
            texture.minFilter = THREE.LinearFilter;
            texture.magFilter = THREE.LinearFilter;
            
            // Dispose of old texture to free memory
            if (backgroundMaterial.uniforms.backgroundTexture.value) {
                backgroundMaterial.uniforms.backgroundTexture.value.dispose();
            }
            
            // Set new texture
            backgroundMaterial.uniforms.backgroundTexture.value = texture;
            backgroundMaterial.needsUpdate = true;
            
            // Update aspect ratios
            updateTextureAspectRatio();
        });
    }
}

// Handle window resize
function onWindowResize() {
    rendererWidth = window.innerWidth;
    rendererHeight = window.innerHeight;
    
    renderer.setSize(rendererWidth, rendererHeight);
    composer.setSize(rendererWidth, rendererHeight);
    
    // Update texture aspect ratios
    updateTextureAspectRatio();
}

// Update texture aspect ratio to maintain proper dimensions
function updateTextureAspectRatio() {
    const screenAspect = rendererWidth / rendererHeight;
    
    // Update video mesh aspect ratio if we have frame dimensions
    if (videoTexture.image) {
        const videoAspect = videoTexture.image.width / videoTexture.image.height;
        
        // Scale the plane to maintain aspect ratio
        if (videoAspect > screenAspect) {
            // Video is wider than screen
            videoMesh.scale.set(1, screenAspect / videoAspect, 1);
        } else {
            // Video is taller than screen
            videoMesh.scale.set(videoAspect / screenAspect, 1, 1);
        }
    }
    
    // Update background mesh aspect ratio if we have background texture
    if (backgroundMaterial.uniforms.backgroundTexture.value && 
        backgroundMaterial.uniforms.backgroundTexture.value.image) {
        const bgTexture = backgroundMaterial.uniforms.backgroundTexture.value;
        const bgAspect = bgTexture.image.width / bgTexture.image.height;
        
        // Scale the background to cover the screen, like CSS background-size: cover
        if (bgAspect > screenAspect) {
            // Background is wider than screen
            backgroundMesh.scale.set(bgAspect / screenAspect, 1, 1);
        } else {
            // Background is taller than screen
            backgroundMesh.scale.set(1, screenAspect / bgAspect, 1);
        }
    }
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    composer.render(); // Use composer instead of renderer directly
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
        try {
            const jsonData = JSON.parse(event.data);
            console.log('Received JSON message:', jsonData);
            
            // Handle source image switching
            if (jsonData.status === 'source_switched') {
                const sourceIndex = jsonData.current_source;
                console.log(`Source image switched to index ${sourceIndex}: ${jsonData.source_name}`);
                showStatus(`Source changed to: ${jsonData.source_name}`, false);
                
                // Switch background to match the source
                switchBackground(sourceIndex);
            }
        } catch (e) {
            // If it's not valid JSON, just log it
            console.log('Received text message:', event.data);
        }
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
            
            // Update the shader uniform
            videoMaterial.uniforms.videoTexture.value = videoTexture;
            
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
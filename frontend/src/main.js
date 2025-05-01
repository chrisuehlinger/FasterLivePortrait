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

// Fluid simulation variables
let fluidCanvas;
let fluidSimulation;
let isTransitioning = false;
let transitionStartTime = 0;
let transitionDuration = 3000; // transition duration in ms
let capturedVideoFrame = null;

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

// Initialize fluid simulation
function initFluidSimulation() {
    // Create canvas for fluid simulation
    fluidCanvas = document.createElement('canvas');
    fluidCanvas.width = window.innerWidth;
    fluidCanvas.height = window.innerHeight;
    fluidCanvas.style.position = 'absolute';
    fluidCanvas.style.top = '0';
    fluidCanvas.style.left = '0';
    fluidCanvas.style.zIndex = '10';
    fluidCanvas.style.pointerEvents = 'none'; // Allow clicks to pass through
    fluidCanvas.style.opacity = '0'; // Hidden by default
    document.body.appendChild(fluidCanvas);
    
    // Initialize fluid simulation with the canvas
    fluidSimulation = new FluidSimulation(fluidCanvas);
    
    // Configure fluid simulation for transitions
    fluidSimulation.config.DENSITY_DISSIPATION = 0.97;
    fluidSimulation.config.VELOCITY_DISSIPATION = 0.98;
    fluidSimulation.config.PRESSURE_DISSIPATION = 0.8;
    fluidSimulation.config.SPLAT_RADIUS = 0.6;
    fluidSimulation.config.SPLAT_FORCE = 6000;
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
    
    // Also resize the fluid simulation canvas if it exists
    if (fluidCanvas) {
        fluidCanvas.width = rendererWidth;
        fluidCanvas.height = rendererHeight;
        fluidSimulation.resize();
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
    
    // Render Three.js scene
    renderer.render(scene, camera);
    
    // Handle fluid transition if active
    if (isTransitioning) {
        const elapsed = performance.now() - transitionStartTime;
        const progress = Math.min(elapsed / transitionDuration, 1.0);
        
        // Update fluid simulation
        fluidSimulation.update();
        
        if (progress < 1.0) {
            // During transition, adjust opacity
            fluidCanvas.style.opacity = (1.0 - progress < 0.1) ? 
                (1.0 - progress) * 10 : // Fade out at the end
                1.0;                     // Full opacity during main transition
            
            // Create random fluid splatters from the image to simulate dispersing
            if (progress < 0.7 && capturedVideoFrame && Math.random() > 0.9) {
                createSplatFromImage(capturedVideoFrame, progress);
            }
        } else {
            // Transition complete
            fluidCanvas.style.opacity = '0';
            isTransitioning = false;
            capturedVideoFrame = null;
        }
    }
}

// Create fluid splats based on the current video frame
function createSplatFromImage(imageBitmap, progress) {
    // Create a temporary canvas to read pixel data
    const tempCanvas = document.createElement('canvas');
    const ctx = tempCanvas.getContext('2d');
    tempCanvas.width = imageBitmap.width;
    tempCanvas.height = imageBitmap.height;
    
    // Draw the image to the canvas
    ctx.drawImage(imageBitmap, 0, 0);
    
    // Get image data
    const imageData = ctx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const data = imageData.data;
    
    // Sample some pixels to create splats
    const numSplats = 1 + Math.floor(progress * 5); // More splats as transition progresses
    
    for (let i = 0; i < numSplats; i++) {
        // Choose a random significant pixel
        let attempts = 0;
        let x, y, r, g, b, brightness;
        
        // Look for pixels with enough brightness/color
        do {
            x = Math.floor(Math.random() * tempCanvas.width);
            y = Math.floor(Math.random() * tempCanvas.height);
            
            const pixelIndex = (y * tempCanvas.width + x) * 4;
            r = data[pixelIndex] / 255;
            g = data[pixelIndex + 1] / 255;
            b = data[pixelIndex + 2] / 255;
            
            brightness = (r + g + b) / 3;
            attempts++;
        } while (brightness < 0.3 && attempts < 20);
        
        // Scale coordinates to match fluid simulation canvas
        const canvasX = (x / tempCanvas.width) * fluidCanvas.width;
        const canvasY = (y / tempCanvas.height) * fluidCanvas.height;
        
        // Random velocity based on position and progress
        const angle = Math.random() * Math.PI * 2;
        const force = 5 + Math.random() * 15 * progress;
        const dx = Math.cos(angle) * force;
        const dy = Math.sin(angle) * force;
        
        // Create fluid splat
        fluidSimulation.splat(canvasX, canvasY, dx, dy, [r, g, b]);
    }
}

// Start a fluid transition effect
function startFluidTransition(currentFrame) {
    if (!fluidSimulation) {
        console.warn("Fluid simulation not initialized");
        return;
    }
    
    // Capture the current frame to use for fluid particles
    capturedVideoFrame = currentFrame;
    
    // Reset fluid simulation
    fluidSimulation.reset();
    
    // Create initial splats based on the image
    for (let i = 0; i < 10; i++) {
        createSplatFromImage(capturedVideoFrame, 0);
    }
    
    // Begin transition
    isTransitioning = true;
    transitionStartTime = performance.now();
    fluidCanvas.style.opacity = '1';
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
                console.log(`Source image switched to index ${jsonData.current_source}: ${jsonData.source_name}`);
                showStatus(`Source changed to: ${jsonData.source_name}`, false);
                
                // Trigger fluid transition with current frame
                if (videoTexture.image && fluidSimulation) {
                    startFluidTransition(videoTexture.image);
                }
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
        if (!videoTexture.image || videoTexture.image !== imageBitmap) {
            // Clean up previous texture data if it exists
            if (videoTexture && videoTexture.image) {
                // Three.js doesn't have a direct dispose method on the image
                // Just mark the texture for update instead of trying to dispose the image
                videoTexture.needsUpdate = true;
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
    
    // Initialize Fluid Simulation
    initFluidSimulation();
    
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
        
        // T or t key to manually trigger fluid transition (for testing)
        if (e.key === 't' || e.key === 'T') {
            if (videoTexture.image && fluidSimulation) {
                startFluidTransition(videoTexture.image);
            }
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


// Fluid Simulation class - complete implementation from WebGL-Fluid-Simulation
class FluidSimulation {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2');
        
        if (!this.gl) {
            this.gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        }
        
        if (!this.gl) {
            console.error('WebGL not supported');
            return;
        }
        
        // Extension support
        this.ext = {
            formatRGBA: { internalFormat: this.gl.RGBA, format: this.gl.RGBA },
            formatRG: null,
            formatR: null,
            halfFloatTexType: null,
            supportLinearFiltering: null
        };
        
        this.isWebGL2 = !!this.gl.getExtension('EXT_color_buffer_float');
        
        if (this.isWebGL2) {
            this.ext.formatRGBA = { internalFormat: this.gl.RGBA16F, format: this.gl.RGBA };
            this.ext.formatRG = { internalFormat: this.gl.RG16F, format: this.gl.RG };
            this.ext.formatR = { internalFormat: this.gl.R16F, format: this.gl.RED };
            this.ext.halfFloatTexType = this.gl.HALF_FLOAT;
            this.ext.supportLinearFiltering = this.gl.getExtension('OES_texture_float_linear');
        } else {
            this.ext.formatRGBA = { internalFormat: this.gl.RGBA, format: this.gl.RGBA };
            this.ext.formatRG = { internalFormat: this.gl.RGBA, format: this.gl.RGBA };
            this.ext.formatR = { internalFormat: this.gl.RGBA, format: this.gl.RGBA };
            
            // Check for half float texture support
            if (this.gl.getExtension('OES_texture_half_float')) {
                this.ext.halfFloatTexType = this.gl.getExtension('OES_texture_half_float').HALF_FLOAT_OES;
                this.ext.supportLinearFiltering = this.gl.getExtension('OES_texture_half_float_linear');
            } else {
                this.ext.halfFloatTexType = this.gl.UNSIGNED_BYTE;
                console.warn('Using 8-bit textures due to lack of half float support');
            }
        }
        
        // Configuration
        this.config = {
            TEXTURE_DOWNSAMPLE: 1,
            DENSITY_DISSIPATION: 0.98,
            VELOCITY_DISSIPATION: 0.99,
            PRESSURE_DISSIPATION: 0.8,
            PRESSURE_ITERATIONS: 20,
            CURL: 30,
            SPLAT_RADIUS: 0.5,
            SPLAT_FORCE: 6000,
            SHADING: true,
            COLORFUL: true,
            PAUSED: false,
            BACK_COLOR: { r: 0, g: 0, b: 0 }
        };
        
        // Initialize shaders, framebuffers, etc.
        this.init();
    }
    
    init() {
        this.initShaders();
        this.initFramebuffers();
        
        // Create a quad that fills the screen
        const gl = this.gl;
        this.blit = (() => {
            // Create vertices for a quad covering the entire viewport
            gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
            gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
            gl.enableVertexAttribArray(0);
            
            return (destination) => {
                gl.bindFramebuffer(gl.FRAMEBUFFER, destination);
                gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
            };
        })();
        
        // Initial resize
        this.resize();
    }
    
    initShaders() {
        const gl = this.gl;
        
        // Basic vertex shader for all programs
        const baseVertexShader = compileShader(gl, gl.VERTEX_SHADER, `
            precision highp float;
            attribute vec2 aPosition;
            varying vec2 vUv;
            varying vec2 vL;
            varying vec2 vR;
            varying vec2 vT;
            varying vec2 vB;
            uniform vec2 texelSize;
            void main () {
                vUv = aPosition * 0.5 + 0.5;
                vL = vUv - vec2(texelSize.x, 0.0);
                vR = vUv + vec2(texelSize.x, 0.0);
                vT = vUv + vec2(0.0, texelSize.y);
                vB = vUv - vec2(0.0, texelSize.y);
                gl_Position = vec4(aPosition, 0.0, 1.0);
            }
        `);
        
        // Clear shader for resetting simulation
        const clearShader = compileShader(gl, gl.FRAGMENT_SHADER, `
            precision mediump float;
            precision mediump sampler2D;
            varying highp vec2 vUv;
            uniform sampler2D uTexture;
            uniform float value;
            void main () {
                gl_FragColor = value * texture2D(uTexture, vUv);
            }
        `);
        
        // Display shader for rendering the fluid simulation
        const displayShader = compileShader(gl, gl.FRAGMENT_SHADER, `
            precision highp float;
            precision highp sampler2D;
            varying vec2 vUv;
            uniform sampler2D uTexture;
            void main () {
                vec3 color = texture2D(uTexture, vUv).rgb;
                gl_FragColor = vec4(color, 1.0);
            }
        `);
        
        // Advection shader for moving fluid based on velocity
        const advectionShader = compileShader(gl, gl.FRAGMENT_SHADER, `
            precision highp float;
            precision highp sampler2D;
            varying vec2 vUv;
            uniform sampler2D uVelocity;
            uniform sampler2D uSource;
            uniform vec2 texelSize;
            uniform float dt;
            uniform float dissipation;
            void main () {
                vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
                gl_FragColor = dissipation * texture2D(uSource, coord);
                gl_FragColor.a = 1.0;
            }
        `);
        
        // Divergence shader for calculating divergence of velocity field
        const divergenceShader = compileShader(gl, gl.FRAGMENT_SHADER, `
            precision mediump float;
            precision mediump sampler2D;
            varying highp vec2 vUv;
            varying highp vec2 vL;
            varying highp vec2 vR;
            varying highp vec2 vT;
            varying highp vec2 vB;
            uniform sampler2D uVelocity;
            void main () {
                float L = texture2D(uVelocity, vL).x;
                float R = texture2D(uVelocity, vR).x;
                float T = texture2D(uVelocity, vT).y;
                float B = texture2D(uVelocity, vB).y;
                vec2 C = texture2D(uVelocity, vUv).xy;
                float div = 0.5 * (R - L + T - B);
                gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
            }
        `);
        
        // Curl shader for calculating vorticity
        const curlShader = compileShader(gl, gl.FRAGMENT_SHADER, `
            precision mediump float;
            precision mediump sampler2D;
            varying highp vec2 vUv;
            varying highp vec2 vL;
            varying highp vec2 vR;
            varying highp vec2 vT;
            varying highp vec2 vB;
            uniform sampler2D uVelocity;
            void main () {
                float L = texture2D(uVelocity, vL).y;
                float R = texture2D(uVelocity, vR).y;
                float T = texture2D(uVelocity, vT).x;
                float B = texture2D(uVelocity, vB).x;
                float vorticity = R - L - T + B;
                gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
            }
        `);
        
        // Vorticity shader for enhancing curl (swirls)
        const vorticityShader = compileShader(gl, gl.FRAGMENT_SHADER, `
            precision highp float;
            precision highp sampler2D;
            varying vec2 vUv;
            varying vec2 vL;
            varying vec2 vR;
            varying vec2 vT;
            varying vec2 vB;
            uniform sampler2D uVelocity;
            uniform sampler2D uCurl;
            uniform float curl;
            uniform float dt;
            void main () {
                float L = texture2D(uCurl, vL).x;
                float R = texture2D(uCurl, vR).x;
                float T = texture2D(uCurl, vT).x;
                float B = texture2D(uCurl, vB).x;
                float C = texture2D(uCurl, vUv).x;
                vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
                force /= length(force) + 0.0001;
                force *= curl * C;
                force.y *= -1.0;
                vec2 velocity = texture2D(uVelocity, vUv).xy;
                velocity += force * dt;
                velocity = min(max(velocity, -1000.0), 1000.0);
                gl_FragColor = vec4(velocity, 0.0, 1.0);
            }
        `);
        
        // Pressure shader for iterative pressure solving
        const pressureShader = compileShader(gl, gl.FRAGMENT_SHADER, `
            precision mediump float;
            precision mediump sampler2D;
            varying highp vec2 vUv;
            varying highp vec2 vL;
            varying highp vec2 vR;
            varying highp vec2 vT;
            varying highp vec2 vB;
            uniform sampler2D uPressure;
            uniform sampler2D uDivergence;
            void main () {
                float L = texture2D(uPressure, vL).x;
                float R = texture2D(uPressure, vR).x;
                float T = texture2D(uPressure, vT).x;
                float B = texture2D(uPressure, vB).x;
                float C = texture2D(uPressure, vUv).x;
                float divergence = texture2D(uDivergence, vUv).x;
                float pressure = (L + R + B + T - divergence) * 0.25;
                gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
            }
        `);
        
        // Gradient subtraction shader for applying pressure gradient to velocity
        const gradientSubtractShader = compileShader(gl, gl.FRAGMENT_SHADER, `
            precision mediump float;
            precision mediump sampler2D;
            varying highp vec2 vUv;
            varying highp vec2 vL;
            varying highp vec2 vR;
            varying highp vec2 vT;
            varying highp vec2 vB;
            uniform sampler2D uPressure;
            uniform sampler2D uVelocity;
            void main () {
                float L = texture2D(uPressure, vL).x;
                float R = texture2D(uPressure, vR).x;
                float T = texture2D(uPressure, vT).x;
                float B = texture2D(uPressure, vB).x;
                vec2 velocity = texture2D(uVelocity, vUv).xy;
                velocity.xy -= vec2(R - L, T - B);
                gl_FragColor = vec4(velocity, 0.0, 1.0);
            }
        `);
        
        // Splat shader for adding "dye" to the fluid
        const splatShader = compileShader(gl, gl.FRAGMENT_SHADER, `
            precision highp float;
            precision highp sampler2D;
            varying vec2 vUv;
            uniform sampler2D uTarget;
            uniform float aspectRatio;
            uniform vec3 color;
            uniform vec2 point;
            uniform float radius;
            void main () {
                vec2 p = vUv - point.xy;
                p.x *= aspectRatio;
                vec3 splat = exp(-dot(p, p) / radius) * color;
                vec3 base = texture2D(uTarget, vUv).xyz;
                gl_FragColor = vec4(base + splat, 1.0);
            }
        `);
        
        // Helper function to compile shaders
        function compileShader(gl, type, source) {
            const shader = gl.createShader(type);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
                gl.deleteShader(shader);
                return null;
            }
            
            return shader;
        }
        
        // Create all shader programs
        this.programs = {};
        this.programs.clear = this.createProgram(baseVertexShader, clearShader);
        this.programs.display = this.createProgram(baseVertexShader, displayShader);
        this.programs.advection = this.createProgram(baseVertexShader, advectionShader);
        this.programs.divergence = this.createProgram(baseVertexShader, divergenceShader);
        this.programs.curl = this.createProgram(baseVertexShader, curlShader);
        this.programs.vorticity = this.createProgram(baseVertexShader, vorticityShader);
        this.programs.pressure = this.createProgram(baseVertexShader, pressureShader);
        this.programs.gradientSubtract = this.createProgram(baseVertexShader, gradientSubtractShader);
        this.programs.splat = this.createProgram(baseVertexShader, splatShader);
    }
    
    createProgram(vertexShader, fragmentShader) {
        const gl = this.gl;
        const program = gl.createProgram();
        
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('Program linking error:', gl.getProgramInfoLog(program));
            gl.deleteProgram(program);
            return null;
        }
        
        return program;
    }
    
    initFramebuffers() {
        const texType = this.ext.halfFloatTexType;
        const gl = this.gl;
        
        // Adjust the resolution based on the TEXTURE_DOWNSAMPLE setting
        this.simWidth = Math.round(this.canvas.width / this.config.TEXTURE_DOWNSAMPLE);
        this.simHeight = Math.round(this.canvas.height / this.config.TEXTURE_DOWNSAMPLE);
        
        // Create the double framebuffers for ping-pong rendering
        this.density = this.createDoubleFBO(this.simWidth, this.simHeight, this.ext.formatRGBA.internalFormat, this.ext.formatRGBA.format, texType, this.ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST);
        this.velocity = this.createDoubleFBO(this.simWidth, this.simHeight, this.ext.formatRG.internalFormat, this.ext.formatRG.format, texType, this.ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST);
        this.divergence = this.createFBO(this.simWidth, this.simHeight, this.ext.formatR.internalFormat, this.ext.formatR.format, texType, gl.NEAREST);
        this.curl = this.createFBO(this.simWidth, this.simHeight, this.ext.formatR.internalFormat, this.ext.formatR.format, texType, gl.NEAREST);
        this.pressure = this.createDoubleFBO(this.simWidth, this.simHeight, this.ext.formatR.internalFormat, this.ext.formatR.format, texType, gl.NEAREST);
    }
    
    createFBO(w, h, internalFormat, format, type, param) {
        const gl = this.gl;
        const fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        
        const texture = this.createTexture(w, h, internalFormat, format, type, param);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
        gl.viewport(0, 0, w, h);
        gl.clear(gl.COLOR_BUFFER_BIT);
        
        return {
            texture,
            fbo,
            width: w,
            height: h,
            attach(id) {
                gl.activeTexture(gl.TEXTURE0 + id);
                gl.bindTexture(gl.TEXTURE_2D, texture);
                return id;
            }
        };
    }
    
    createDoubleFBO(w, h, internalFormat, format, type, param) {
        return {
            width: w,
            height: h,
            read: this.createFBO(w, h, internalFormat, format, type, param),
            write: this.createFBO(w, h, internalFormat, format, type, param),
            swap() {
                const temp = this.read;
                this.read = this.write;
                this.write = temp;
            }
        };
    }
    
    createTexture(w, h, internalFormat, format, type, param) {
        const gl = this.gl;
        gl.activeTexture(gl.TEXTURE0);
        
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
        
        return texture;
    }
    
    update() {
        if (this.config.PAUSED) return;
        
        const gl = this.gl;
        const dt = 0.016; // Assuming ~60fps
        
        // Apply advection to velocity and density
        this.advect(this.velocity.read, this.velocity.read, this.velocity.write, this.config.VELOCITY_DISSIPATION);
        this.velocity.swap();
        
        this.advect(this.velocity.read, this.density.read, this.density.write, this.config.DENSITY_DISSIPATION);
        this.density.swap();
        
        // Calculate curl and apply vorticity confinement
        this.curl(this.velocity.read, this.curl);
        this.vorticity(this.velocity.read, this.curl, this.velocity.write);
        this.velocity.swap();
        
        // Calculate divergence and solve pressure
        this.divergence(this.velocity.read, this.divergence);
        this.clear(this.pressure.read);
        this.clear(this.pressure.write);
        
        // Pressure iterations
        for (let i = 0; i < this.config.PRESSURE_ITERATIONS; i++) {
            this.pressure(this.pressure.read, this.divergence, this.pressure.write);
            this.pressure.swap();
        }
        
        // Apply pressure gradient to velocity
        this.gradientSubtract(this.velocity.read, this.pressure.read, this.velocity.write);
        this.velocity.swap();
        
        // Render the result
        this.render();
    }
    
    advect(velocity, source, dest, dissipation) {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, dest.fbo);
        
        // Bind the advection program
        gl.useProgram(this.programs.advection);
        
        // Set texel size for neighboring cell access
        gl.uniform2f(gl.getUniformLocation(this.programs.advection, 'texelSize'), 1.0 / velocity.width, 1.0 / velocity.height);
        
        // Set time step
        gl.uniform1f(gl.getUniformLocation(this.programs.advection, 'dt'), 0.016);
        
        // Set dissipation rate
        gl.uniform1f(gl.getUniformLocation(this.programs.advection, 'dissipation'), dissipation);
        
        // Bind textures
        gl.uniform1i(gl.getUniformLocation(this.programs.advection, 'uVelocity'), velocity.attach(0));
        gl.uniform1i(gl.getUniformLocation(this.programs.advection, 'uSource'), source.attach(1));
        
        // Draw
        this.blit(dest.fbo);
        
        // Unbind framebuffer
        velocity.detach(0);
        source.detach(1);
    }
    
    curl(velocity, dest) {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, dest.fbo);
        
        // Bind the curl program
        gl.useProgram(this.programs.curl);
        
        // Set texel size for neighboring cell access
        gl.uniform2f(gl.getUniformLocation(this.programs.curl, 'texelSize'), 1.0 / velocity.width, 1.0 / velocity.height);
        
        // Bind velocity texture
        gl.uniform1i(gl.getUniformLocation(this.programs.curl, 'uVelocity'), velocity.attach(0));
        
        // Draw
        this.blit(dest.fbo);
        velocity.detach(0);
    }
    
    vorticity(velocity, curl, dest) {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, dest.fbo);
        
        // Bind the vorticity program
        gl.useProgram(this.programs.vorticity);
        
        // Set texel size for neighboring cell access
        gl.uniform2f(gl.getUniformLocation(this.programs.vorticity, 'texelSize'), 1.0 / velocity.width, 1.0 / velocity.height);
        
        // Set curl strength and time step
        gl.uniform1f(gl.getUniformLocation(this.programs.vorticity, 'curl'), this.config.CURL);
        gl.uniform1f(gl.getUniformLocation(this.programs.vorticity, 'dt'), 0.016);
        
        // Bind textures
        gl.uniform1i(gl.getUniformLocation(this.programs.vorticity, 'uVelocity'), velocity.attach(0));
        gl.uniform1i(gl.getUniformLocation(this.programs.vorticity, 'uCurl'), curl.attach(1));
        
        // Draw
        this.blit(dest.fbo);
        velocity.detach(0);
        curl.detach(1);
    }
    
    divergence(velocity, dest) {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, dest.fbo);
        
        // Bind the divergence program
        gl.useProgram(this.programs.divergence);
        
        // Set texel size for neighboring cell access
        gl.uniform2f(gl.getUniformLocation(this.programs.divergence, 'texelSize'), 1.0 / velocity.width, 1.0 / velocity.height);
        
        // Bind velocity texture
        gl.uniform1i(gl.getUniformLocation(this.programs.divergence, 'uVelocity'), velocity.attach(0));
        
        // Draw
        this.blit(dest.fbo);
        velocity.detach(0);
    }
    
    pressure(pressure, divergence, dest) {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, dest.fbo);
        
        // Bind the pressure program
        gl.useProgram(this.programs.pressure);
        
        // Set texel size for neighboring cell access
        gl.uniform2f(gl.getUniformLocation(this.programs.pressure, 'texelSize'), 1.0 / pressure.width, 1.0 / pressure.height);
        
        // Bind textures
        gl.uniform1i(gl.getUniformLocation(this.programs.pressure, 'uPressure'), pressure.attach(0));
        gl.uniform1i(gl.getUniformLocation(this.programs.pressure, 'uDivergence'), divergence.attach(1));
        
        // Draw
        this.blit(dest.fbo);
        pressure.detach(0);
        divergence.detach(1);
    }
    
    gradientSubtract(velocity, pressure, dest) {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, dest.fbo);
        
        // Bind the gradient subtraction program
        gl.useProgram(this.programs.gradientSubtract);
        
        // Set texel size for neighboring cell access
        gl.uniform2f(gl.getUniformLocation(this.programs.gradientSubtract, 'texelSize'), 1.0 / velocity.width, 1.0 / velocity.height);
        
        // Bind textures
        gl.uniform1i(gl.getUniformLocation(this.programs.gradientSubtract, 'uPressure'), pressure.attach(0));
        gl.uniform1i(gl.getUniformLocation(this.programs.gradientSubtract, 'uVelocity'), velocity.attach(1));
        
        // Draw
        this.blit(dest.fbo);
        pressure.detach(0);
        velocity.detach(1);
    }
    
    splat(x, y, dx, dy, color) {
        const gl = this.gl;
        
        // Apply to velocity
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.velocity.write.fbo);
        gl.useProgram(this.programs.splat);
        
        gl.uniform1i(gl.getUniformLocation(this.programs.splat, 'uTarget'), this.velocity.read.attach(0));
        gl.uniform1f(gl.getUniformLocation(this.programs.splat, 'aspectRatio'), this.canvas.width / this.canvas.height);
        gl.uniform2f(gl.getUniformLocation(this.programs.splat, 'point'), x / this.canvas.width, 1.0 - y / this.canvas.height);
        gl.uniform3f(gl.getUniformLocation(this.programs.splat, 'color'), dx, -dy, 1.0);
        gl.uniform1f(gl.getUniformLocation(this.programs.splat, 'radius'), this.config.SPLAT_RADIUS / 100.0);
        
        this.blit(this.velocity.write.fbo);
        this.velocity.swap();
        
        // Apply to density (color)
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.density.write.fbo);
        gl.useProgram(this.programs.splat);
        
        gl.uniform1i(gl.getUniformLocation(this.programs.splat, 'uTarget'), this.density.read.attach(0));
        gl.uniform1f(gl.getUniformLocation(this.programs.splat, 'aspectRatio'), this.canvas.width / this.canvas.height);
        gl.uniform2f(gl.getUniformLocation(this.programs.splat, 'point'), x / this.canvas.width, 1.0 - y / this.canvas.height);
        gl.uniform3f(gl.getUniformLocation(this.programs.splat, 'color'), color[0], color[1], color[2]);
        gl.uniform1f(gl.getUniformLocation(this.programs.splat, 'radius'), this.config.SPLAT_RADIUS / 100.0);
        
        this.blit(this.density.write.fbo);
        this.density.swap();
    }
    
    render() {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        
        // Set clear color based on config
        const { r, g, b } = this.config.BACK_COLOR;
        gl.clearColor(r / 255, g / 255, b / 255, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        
        // Render density to screen
        gl.useProgram(this.programs.display);
        gl.uniform1i(gl.getUniformLocation(this.programs.display, 'uTexture'), this.density.read.attach(0));
        this.blit(null);
        this.density.read.detach(0);
    }
    
    clear(target) {
        const gl = this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
        gl.clearColor(0.0, 0.0, 0.0, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);
    }
    
    reset() {
        // Clear all framebuffers
        this.clear(this.velocity.read);
        this.clear(this.velocity.write);
        this.clear(this.density.read);
        this.clear(this.density.write);
        this.clear(this.pressure.read);
        this.clear(this.pressure.write);
        this.clear(this.divergence);
        this.clear(this.curl);
    }
    
    resize() {
        const gl = this.gl;
        
        // Only resize if dimensions have changed
        const width = gl.canvas.width;
        const height = gl.canvas.height;
        
        if (this.canvas.width !== width || this.canvas.height !== height) {
            this.canvas.width = width;
            this.canvas.height = height;
        }
        
        // Re-initialize framebuffers with new dimensions
        this.initFramebuffers();
    }
}

Object.defineProperty(FluidSimulation.prototype, 'FRAMEBUFFER_UNSUPPORTED', { get: function() {
    return this.gl.checkFramebufferStatus(this.gl.FRAMEBUFFER) !== this.gl.FRAMEBUFFER_COMPLETE; 
}});

// Helper methods for attach/detach
Object.defineProperty(FluidSimulation.prototype.createFBO.prototype, 'detach', {
    value: function(id) {
        const gl = this.gl;
        gl.activeTexture(gl.TEXTURE0 + id);
        gl.bindTexture(gl.TEXTURE_2D, null);
        return id;
    }
});

// Initialize the application
initialize();

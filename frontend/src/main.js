// Import Three.js
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
// websocket-handler.js is imported in viewer.html and exposed via window

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
let perlinNoisePass; // New variable for Perlin noise pass
let bloomPass; // Variable for UnrealBloom pass
let clock; // Clock for time tracking

// Transition effect variables
let deepBackgroundTexture, deepBackgroundMaterial, deepBackgroundMesh;
let transitionRenderTarget;
let isTransitioning = false;
let transitionStartTime = 0;

// Three.js variables
let scene, camera, renderer;
let videoTexture, videoMaterial, videoMesh;
let backgroundTexture, backgroundMaterial, backgroundMesh;
let composer;
let rendererWidth = window.innerWidth;
let rendererHeight = window.innerHeight;

const textureLoader = new THREE.TextureLoader();

// Background images corresponding to each source
const backgroundTextures = [
    '/frontend/images/ahau-kin-bg.png',
    '/frontend/images/ix-chel-bg.png',
    '/frontend/images/chac-bolay-bg.png',
    '/frontend/images/ahau-kin-bg.png',
].map(path => {
    return textureLoader.load(path, (texture) => {
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
    });
});
const keyColors = [
    new THREE.Color('yellow'),  // Placeholder paths - update these later
    new THREE.Color('blue'),
    new THREE.Color('red'),
    new THREE.Color('black')
];
const keyLuminosities = keyColors.map(color => {
    const luminance = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
    return luminance;
});

const characterScales = [
    0.85,  
    0.5,
    0.9,
    0.1,  
];
const characterTopOffsets = [
    1,  
    0.75,
    0,
    0,
];

let animationIntensity = 0;
let currentBackgroundIndex = 3;

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
    
    // Create render target for transition effects
    transitionRenderTarget = new THREE.WebGLRenderTarget(
        window.innerWidth, 
        window.innerHeight, 
        {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
            format: THREE.RGBAFormat
        }
    );
    
    // Create deep background layer (rendered behind everything)
    deepBackgroundMaterial = new THREE.ShaderMaterial({
        uniforms: {
            'capturedTexture': { value: null },
            'time': { value: 0.0 },
            'displacementScale': { value: 0.0 },
            'turbulenceFrequency': { value: 5.0 },
            'opacity': { value: 0.0 }
        },
        vertexShader: `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            uniform sampler2D capturedTexture;
            uniform float time;
            uniform float displacementScale;
            uniform float turbulenceFrequency;
            uniform float opacity;
            varying vec2 vUv;
            
            // Perlin noise functions
            vec3 fade(vec3 t) {
                return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
            }

            vec4 permute(vec4 x) {
                return mod(((x*34.0)+1.0)*x, 289.0);
            }

            float cnoise(vec3 P) {
                vec3 Pi0 = floor(P);
                vec3 Pi1 = Pi0 + vec3(1.0);
                Pi0 = mod(Pi0, 289.0);
                Pi1 = mod(Pi1, 289.0);
                vec3 Pf0 = fract(P);
                vec3 Pf1 = Pf0 - vec3(1.0);
                vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
                vec4 iy = vec4(Pi0.y, Pi0.y, Pi1.y, Pi1.y);
                vec4 iz0 = Pi0.z * vec4(1.0);
                vec4 iz1 = Pi1.z * vec4(1.0);

                vec4 ixy = permute(permute(ix) + iy);
                vec4 ixy0 = permute(ixy + iz0);
                vec4 ixy1 = permute(ixy + iz1);

                vec4 gx0 = ixy0 / 7.0;
                vec4 gy0 = fract(floor(gx0) / 7.0) - 0.5;
                gx0 = fract(gx0);
                vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
                vec4 sz0 = step(gz0, vec4(0.0));
                gx0 -= sz0 * (step(0.0, gx0) - 0.5);
                gy0 -= sz0 * (step(0.0, gy0) - 0.5);

                vec4 gx1 = ixy1 / 7.0;
                vec4 gy1 = fract(floor(gx1) / 7.0) - 0.5;
                gx1 = fract(gx1);
                vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
                vec4 sz1 = step(gz1, vec4(0.0));
                gx1 -= sz1 * (step(0.0, gx1) - 0.5);
                gy1 -= sz1 * (step(0.0, gy1) - 0.5);

                vec3 g000 = vec3(gx0.x, gy0.x, gz0.x);
                vec3 g100 = vec3(gx0.y, gy0.y, gz0.y);
                vec3 g010 = vec3(gx0.z, gy0.z, gz0.z);
                vec3 g110 = vec3(gx0.w, gy0.w, gz0.w);
                vec3 g001 = vec3(gx1.x, gy1.x, gz1.x);
                vec3 g101 = vec3(gx1.y, gy1.y, gz1.y);
                vec3 g011 = vec3(gx1.z, gy1.z, gz1.z);
                vec3 g111 = vec3(gx1.w, gy1.w, gz1.w);

                vec4 norm0 = inversesqrt(vec4(dot(g000, g000), dot(g100, g100), dot(g010, g010), dot(g110, g110)));
                g000 *= norm0.x;
                g100 *= norm0.y;
                g010 *= norm0.z;
                g110 *= norm0.w;
                vec4 norm1 = inversesqrt(vec4(dot(g001, g001), dot(g101, g101), dot(g011, g011), dot(g111, g111)));
                g001 *= norm1.x;
                g101 *= norm1.y;
                g011 *= norm1.z;
                g111 *= norm1.w;

                float n000 = dot(g000, Pf0);
                float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
                float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
                float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
                float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
                float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
                float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
                float n111 = dot(g111, Pf1);

                vec3 fade_xyz = fade(Pf0);
                vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
                vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
                float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
                return 2.2 * n_xyz;
            }
            
            void main() {
                // Apply stronger turbulent displacement for transition effect
                float noiseX = cnoise(vec3(vUv.x * turbulenceFrequency, vUv.y * turbulenceFrequency, time * 0.4));
                float noiseY = cnoise(vec3(vUv.y * turbulenceFrequency, vUv.x * turbulenceFrequency, time * 0.4 + 100.0));
                
                // Create turbulent, swirling displacement that increases with time
                vec2 displacedUv = vUv;
                displacedUv.x += displacementScale * noiseX;
                displacedUv.y += displacementScale * noiseY;
                
                // Sample texture with displaced coordinates
                vec4 texColor = texture2D(capturedTexture, displacedUv);
                
                // Apply opacity fade
                gl_FragColor = vec4(texColor.rgb, texColor.a * opacity);
            }
        `,
        transparent: true,
        side: THREE.FrontSide
    });
    
    const deepBackgroundGeometry = new THREE.PlaneGeometry(2, 2);
    deepBackgroundMesh = new THREE.Mesh(deepBackgroundGeometry, deepBackgroundMaterial);
    deepBackgroundMesh.position.z = -0.2; // Place behind everything else
    scene.add(deepBackgroundMesh);
    
    // Create background layer first (rendered behind)
    backgroundTexture = backgroundTextures[0];
    
    // Create inverted background material using shader material
    backgroundMaterial = new THREE.ShaderMaterial({
        uniforms: {
            'backgroundTexture': { value: backgroundTexture },
            'threshold': { value: 0.05 },
            'keyColor': { value: keyColors[currentBackgroundIndex] },
            'time': { value: 0.0 },                     // New: Time for animation
            'displacementScale': { value: 0.015 },       // New: How strong the displacement is
            'turbulenceFrequency': { value: 1.5 }       // New: Frequency of the turbulence
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
            uniform float time;
            uniform float displacementScale;
            uniform float turbulenceFrequency;
            varying vec2 vUv;
            
            // Perlin noise functions
            vec3 fade(vec3 t) {
                return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
            }

            vec4 permute(vec4 x) {
                return mod(((x*34.0)+1.0)*x, 289.0);
            }

            float cnoise(vec3 P) {
                vec3 Pi0 = floor(P);
                vec3 Pi1 = Pi0 + vec3(1.0);
                Pi0 = mod(Pi0, 289.0);
                Pi1 = mod(Pi1, 289.0);
                vec3 Pf0 = fract(P);
                vec3 Pf1 = Pf0 - vec3(1.0);
                vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
                vec4 iy = vec4(Pi0.y, Pi0.y, Pi1.y, Pi1.y);
                vec4 iz0 = Pi0.z * vec4(1.0);
                vec4 iz1 = Pi1.z * vec4(1.0);

                vec4 ixy = permute(permute(ix) + iy);
                vec4 ixy0 = permute(ixy + iz0);
                vec4 ixy1 = permute(ixy + iz1);

                vec4 gx0 = ixy0 / 7.0;
                vec4 gy0 = fract(floor(gx0) / 7.0) - 0.5;
                gx0 = fract(gx0);
                vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
                vec4 sz0 = step(gz0, vec4(0.0));
                gx0 -= sz0 * (step(0.0, gx0) - 0.5);
                gy0 -= sz0 * (step(0.0, gy0) - 0.5);

                vec4 gx1 = ixy1 / 7.0;
                vec4 gy1 = fract(floor(gx1) / 7.0) - 0.5;
                gx1 = fract(gx1);
                vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
                vec4 sz1 = step(gz1, vec4(0.0));
                gx1 -= sz1 * (step(0.0, gx1) - 0.5);
                gy1 -= sz1 * (step(0.0, gy1) - 0.5);

                vec3 g000 = vec3(gx0.x, gy0.x, gz0.x);
                vec3 g100 = vec3(gx0.y, gy0.y, gz0.y);
                vec3 g010 = vec3(gx0.z, gy0.z, gz0.z);
                vec3 g110 = vec3(gx0.w, gy0.w, gz0.w);
                vec3 g001 = vec3(gx1.x, gy1.x, gz1.x);
                vec3 g101 = vec3(gx1.y, gy1.y, gz1.y);
                vec3 g011 = vec3(gx1.z, gy1.z, gz1.z);
                vec3 g111 = vec3(gx1.w, gy1.w, gz1.w);

                vec4 norm0 = inversesqrt(vec4(dot(g000, g000), dot(g100, g100), dot(g010, g010), dot(g110, g110)));
                g000 *= norm0.x;
                g100 *= norm0.y;
                g010 *= norm0.z;
                g110 *= norm0.w;
                vec4 norm1 = inversesqrt(vec4(dot(g001, g001), dot(g101, g101), dot(g011, g011), dot(g111, g111)));
                g001 *= norm1.x;
                g101 *= norm1.y;
                g011 *= norm1.z;
                g111 *= norm1.w;

                float n000 = dot(g000, Pf0);
                float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
                float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
                float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
                float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
                float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
                float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
                float n111 = dot(g111, Pf1);

                vec3 fade_xyz = fade(Pf0);
                vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
                vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
                float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
                return 2.2 * n_xyz;
            }
            
            void main() {
                // Apply turbulent displacement
                float noiseX = cnoise(vec3(vUv.x * turbulenceFrequency, vUv.y * turbulenceFrequency, time * 0.3));
                float noiseY = cnoise(vec3(vUv.y * turbulenceFrequency, vUv.x * turbulenceFrequency, time * 0.3 + 100.0));
                
                // Create turbulent, swirling displacement
                vec2 displacedUv = vUv;
                displacedUv.x += displacementScale * noiseX;
                displacedUv.y += displacementScale * noiseY;
                
                // Sample texture with displaced coordinates
                vec4 texColor = texture2D(backgroundTexture, displacedUv);
                
                // Invert the colors
                vec3 inverted = 1.0 - texColor.rgb;

                float luminance = 0.299 * inverted.r + 0.587 * inverted.g + 0.114 * inverted.b;
                luminance = luminance * texColor.a;
                if (luminance < threshold) {
                    gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0); // Transparent
                } else {
                    gl_FragColor = vec4(keyColor, texColor.a);
                }
            }
        `,
        transparent: true,
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
            uniform sampler2D videoTexture;
            uniform float threshold;
            uniform vec3 keyColor;
            varying vec2 vUv;
            
            void main() {
                // Flip the Y coordinate to fix upside-down video
                vec2 flippedUv = vec2(vUv.x, 1.0 - vUv.y);
                vec4 color = texture2D(videoTexture, flippedUv);

                if(color.g > 0.75 && color.b < 0.25 && color.r < 0.25) {
                    gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0); // Black
                    return;
                }
                // Invert the colors
                vec3 inverted = 1.0 - color.rgb;

                float luminance = 0.299 * inverted.r + 0.587 * inverted.g + 0.114 * inverted.b;
                
                // if (luminance < threshold) {
                //     gl_FragColor = vec4(inverted.rgb, 0.0); // Transparent
                // } else {
                    gl_FragColor = vec4(keyColor*luminance, luminance*color.a);
                // }
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

    // Setup clock for time-based animations
    clock = new THREE.Clock();
}

// Function to capture the current scene for transition
function captureCurrentFrame() {
    // Hide the deep background temporarily to avoid feedback loop
    const wasVisible = deepBackgroundMesh.visible;
    deepBackgroundMesh.visible = false;
    
    // Render current scene to the render target
    renderer.setRenderTarget(transitionRenderTarget);
    renderer.render(scene, camera);
    
    // Reset render target
    renderer.setRenderTarget(null);
    
    // Update deep background with captured frame
    deepBackgroundMaterial.uniforms.capturedTexture.value = transitionRenderTarget.texture;
    
    // Set initial opacity to full
    deepBackgroundMaterial.uniforms.opacity.value = 1.0;
    
    // Reset displacement scale
    deepBackgroundMaterial.uniforms.displacementScale.value = 0.01;
    
    // Restore deep background visibility
    deepBackgroundMesh.visible = true;
}

// Set up post-processing pipeline
function setupPostProcessing() {
    // Create composer
    composer = new EffectComposer(renderer);
    
    // Add render pass
    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);
    
    // Add UnrealBloomPass for glow effect
    const bloomParams = {
        strength: 0,    // bloom intensity
        radius: 0.5,      // blur radius
        threshold: 0.1    // luminance threshold - smaller values = more bloom
    };
    
    bloomPass = new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        bloomParams.strength,
        bloomParams.radius,
        bloomParams.threshold
    );
    composer.addPass(bloomPass);
    
    // Add Perlin noise brightness shader pass
    const perlinNoiseBrightness = {
        uniforms: {
            "tDiffuse": { value: null },
            "time": { value: 0.0 },
            "noiseScale": { value: 10.0 },
            "noiseIntensity": { value: 10 },
            "brightnessSpeed": { value: 0.5 },
            "keyIntensity": { value: 1-keyLuminosities[currentBackgroundIndex] },
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
            uniform float time;
            uniform float noiseScale;
            uniform float noiseIntensity;
            uniform float brightnessSpeed;
            uniform float keyIntensity;
            varying vec2 vUv;

            // Classic Perlin 3D Noise implementation
            // Credit: Stefan Gustavson, Linköping University, Sweden
            vec3 fade(vec3 t) {
                return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
            }

            vec4 permute(vec4 x) {
                return mod(((x*34.0)+1.0)*x, 289.0);
            }

            float cnoise(vec3 P) {
                vec3 Pi0 = floor(P);
                vec3 Pi1 = Pi0 + vec3(1.0);
                Pi0 = mod(Pi0, 289.0);
                Pi1 = mod(Pi1, 289.0);
                vec3 Pf0 = fract(P);
                vec3 Pf1 = Pf0 - vec3(1.0);
                vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
                vec4 iy = vec4(Pi0.y, Pi0.y, Pi1.y, Pi1.y);
                vec4 iz0 = Pi0.z * vec4(1.0);
                vec4 iz1 = Pi1.z * vec4(1.0);

                vec4 ixy = permute(permute(ix) + iy);
                vec4 ixy0 = permute(ixy + iz0);
                vec4 ixy1 = permute(ixy + iz1);

                vec4 gx0 = ixy0 / 7.0;
                vec4 gy0 = fract(floor(gx0) / 7.0) - 0.5;
                gx0 = fract(gx0);
                vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
                vec4 sz0 = step(gz0, vec4(0.0));
                gx0 -= sz0 * (step(0.0, gx0) - 0.5);
                gy0 -= sz0 * (step(0.0, gy0) - 0.5);

                vec4 gx1 = ixy1 / 7.0;
                vec4 gy1 = fract(floor(gx1) / 7.0) - 0.5;
                gx1 = fract(gx1);
                vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
                vec4 sz1 = step(gz1, vec4(0.0));
                gx1 -= sz1 * (step(0.0, gx1) - 0.5);
                gy1 -= sz1 * (step(0.0, gy1) - 0.5);

                vec3 g000 = vec3(gx0.x, gy0.x, gz0.x);
                vec3 g100 = vec3(gx0.y, gy0.y, gz0.y);
                vec3 g010 = vec3(gx0.z, gy0.z, gz0.z);
                vec3 g110 = vec3(gx0.w, gy0.w, gz0.w);
                vec3 g001 = vec3(gx1.x, gy1.x, gz1.x);
                vec3 g101 = vec3(gx1.y, gy1.y, gz1.y);
                vec3 g011 = vec3(gx1.z, gy1.z, gz1.z);
                vec3 g111 = vec3(gx1.w, gy1.w, gz1.w);

                vec4 norm0 = inversesqrt(vec4(dot(g000, g000), dot(g100, g100), dot(g010, g010), dot(g110, g110)));
                g000 *= norm0.x;
                g100 *= norm0.y;
                g010 *= norm0.z;
                g110 *= norm0.w;
                vec4 norm1 = inversesqrt(vec4(dot(g001, g001), dot(g101, g101), dot(g011, g011), dot(g111, g111)));
                g001 *= norm1.x;
                g101 *= norm1.y;
                g011 *= norm1.z;
                g111 *= norm1.w;

                float n000 = dot(g000, Pf0);
                float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
                float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
                float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
                float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
                float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
                float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
                float n111 = dot(g111, Pf1);

                vec3 fade_xyz = fade(Pf0);
                vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
                vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
                float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
                return 2.2 * n_xyz;
            }

            void main() {
                // Get current pixel color
                vec4 color = texture2D(tDiffuse, vUv);
                
                // Generate Perlin noise with evolving time
                float noise = cnoise(vec3(vUv * noiseScale, time * brightnessSpeed));
                
                // Normalize noise to 0.0 - 1.0 range and adjust intensity
                noise = (noise + 1.0) * 0.5;
                noise = noise * noiseIntensity;
                noise = noise * keyIntensity;
                
                // Apply noise as brightness variation
                float luminance = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
                color.rgb += noise*luminance;
                
                gl_FragColor = vec4(vec3(noise),1.0);
                gl_FragColor = vec4(color.rgb, 1.0);
            }
        `
    };
    
    perlinNoisePass = new ShaderPass(perlinNoiseBrightness);
    perlinNoisePass.renderToScreen = true;
    composer.addPass(perlinNoisePass);
}

// Switch to a different background image
function switchBackground(index) {
    if (index < backgroundTextures.length) {
        // Capture current frame before switching
        captureCurrentFrame();
        
        // Start transition
        isTransitioning = true;
        transitionStartTime = clock.getElapsedTime();
        
        // Update to new background
        currentBackgroundIndex = index;
        backgroundMaterial.uniforms.backgroundTexture.value = backgroundTextures[index];
        backgroundMaterial.uniforms.keyColor.value = keyColors[index];
        backgroundMaterial.needsUpdate = true;
        videoMaterial.uniforms.keyColor.value = keyColors[index];
        videoMaterial.needsUpdate = true;
        perlinNoisePass.uniforms.keyIntensity.value = 1-keyLuminosities[index];
        
        updateTextureAspectRatio();
    }
}

// Handle window resize
function onWindowResize() {
    rendererWidth = window.innerWidth;
    rendererHeight = window.innerHeight;
    
    renderer.setSize(rendererWidth, rendererHeight);
    composer.setSize(rendererWidth, rendererHeight);
    
    // Update bloom pass resolution if it exists
    if (bloomPass) {
        bloomPass.resolution.set(rendererWidth, rendererHeight);
    }
    
    // Update transition render target size
    if (transitionRenderTarget) {
        transitionRenderTarget.setSize(rendererWidth, rendererHeight);
    }
    
    // Update texture aspect ratios
    updateTextureAspectRatio();
}

// Update texture aspect ratio to maintain proper dimensions
function updateTextureAspectRatio() {
    const screenAspect = rendererWidth / rendererHeight;
    
    // First update background mesh aspect ratio if we have background texture
    let bgScaleX = 1;
    let bgScaleY = 1;
    
    if (backgroundMaterial.uniforms.backgroundTexture.value && 
        backgroundMaterial.uniforms.backgroundTexture.value.image) {
        const bgTexture = backgroundMaterial.uniforms.backgroundTexture.value;
        const bgAspect = bgTexture.image.width / bgTexture.image.height;
        
        // Scale the background to contain within the screen, like CSS background-size: contain
        if (bgAspect > screenAspect) {
            // Background is wider than screen
            bgScaleX = 1;
            bgScaleY = screenAspect / bgAspect;
        } else {
            // Background is taller than screen
            bgScaleX = bgAspect / screenAspect;
            bgScaleY = 1;
        }
        
        backgroundMesh.scale.set(bgScaleX, bgScaleY, 1);
    }
    
    // Now update video mesh aspect ratio if we have frame dimensions
    if (videoTexture.image) {
        // Force a 1:1 aspect ratio for the video regardless of source dimensions
        
        // Calculate the target size for the video (square)
        const targetHeight = characterScales[currentBackgroundIndex]; // Video takes up 60% of the normalized height
        
        // To maintain a true 1:1 aspect ratio (square) regardless of window dimensions:
        // 1. Start with the desired height in normalized coordinates
        const videoScaleY = targetHeight * bgScaleY;
        
        // 2. For a perfect square, we need to account for the screen aspect ratio
        const videoScaleX = videoScaleY * (rendererHeight / rendererWidth);
        
        // Scale the video maintaining 1:1 aspect ratio
        videoMesh.scale.set(videoScaleX, videoScaleY, 1);
        
        // Position the video at the top center of the composition
        videoMesh.position.y = characterTopOffsets[currentBackgroundIndex] * (bgScaleY - videoScaleY);
        videoMesh.position.x = 0; // Center horizontally
    }
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    
    // Update time uniforms for animations
    const currentTime = clock.getElapsedTime();
    
    // Update the time uniform for the Perlin noise shader
    if (perlinNoisePass) {
        perlinNoisePass.uniforms.time.value = currentTime;
    }

    // Adjust bloom parameters based on the source
    if (bloomPass) {
        // Customize bloom for each character if needed
        const bloomStrengths = [0, 2*animationIntensity, 0, 0]; // Example values for different sources
        bloomPass.strength = bloomStrengths[currentBackgroundIndex];
    }
    
    // Update background turbulent displacement time
    if (backgroundMaterial && backgroundMaterial.uniforms.time) {
        backgroundMaterial.uniforms.time.value = currentTime;
    }
    
    // Handle transition effect
    if (isTransitioning) {
        const elapsedSinceTransition = currentTime - transitionStartTime;
        
        // Update deep background uniforms
        deepBackgroundMaterial.uniforms.time.value = currentTime;
        
        // Increase displacement scale based on time since transition started
        deepBackgroundMaterial.uniforms.displacementScale.value = 1.0*elapsedSinceTransition;
        
        // Decrease opacity over time (0.5 per second)
        const newOpacity = 1.0 - (1.0 * elapsedSinceTransition);
        deepBackgroundMaterial.uniforms.opacity.value = newOpacity;
        
        // End transition when opacity reaches zero
        if (newOpacity <= 0) {
            isTransitioning = false;
        }
    }
    
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
    
    const ws = new WebSocket(`${WS_URL}/ws/viewer/${sessionId}`);
    
    console.log(`Setting up WebSocket handler for viewer at ${WS_URL}/ws/viewer/${sessionId}`);
    // Use the websocket handler to manage the connection
    websocket = window.setupWebSocketHandler(ws, {
        onBinaryData: async (data) => {
            try {
                // Handle binary data (frames)
                const blob = data;
                
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
        },
        onIntensityUpdate: (intensity) => {
            animationIntensity = intensity;
            console.log(`Animation intensity updated to: ${animationIntensity}`);
            // Update any animations that use intensity
        }
    });
    
    // Set up explicit event listener for source switching
    document.addEventListener('websocket:source_switched', (e) => {
        const sourceIndex = e.detail.sourceIndex;
        const sourceName = e.detail.sourceName;
        console.log(`Source image switched to index ${sourceIndex}: ${sourceName}`);
        showStatus(`Source changed to: ${sourceName}`, false);
        
        // Switch background to match the source
        switchBackground(sourceIndex);
    });
    
    // Set up event listeners for websocket events
    document.addEventListener('websocket:open', () => {
        isConnected = true;
        showStatus('Connected to stream. Waiting for video...', false);
    });
    
    // Event listeners
    document.addEventListener('websocket:close', (e) => {
        isConnected = false;
        if (e.detail.wasClean) {
            showStatus(`Connection closed: ${e.detail.reason}`, true);
        } else {
            showStatus('Connection lost. Attempting to reconnect...', true);
            setTimeout(connectToWebSocket, 3000);
        }
    });

    document.addEventListener('websocket:error', (e) => {
        console.error('WebSocket error:', e.detail.error);
        showStatus('Connection error. Please try again later.', true);
    });

    // Add handler for generic messages
    document.addEventListener('websocket:message', (e) => {
        console.log('Received generic message:', e.detail);
    });
    
    // Add handler for set_motal messages
    document.addEventListener('websocket:set_motal', (e) => {
        console.log('Received set_motal message from WebSocket:', e.detail);
    });
    
    // Add handler for set_motal messages
    document.addEventListener('websocket:set_motal', (e) => {
        console.log('Received set_motal message:', e.detail);
        // The actual handling of this message is done in viewer.html
    });
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
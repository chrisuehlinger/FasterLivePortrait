import * as THREE from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import addNoiseShaderSource from "./shaders/addNoiseShader.glsl?raw";
import advectionManualFilteringShaderSource from "./shaders/advectionManualFilteringShader.glsl?raw";
import advectionShaderSource from "./shaders/advectionShader.glsl?raw";
import baseVertexShaderSource from "./shaders/baseVertexShader.glsl?raw";
import buoyancyShaderSource from "./shaders/buoyancyShader.glsl?raw";
import clearShaderSource from "./shaders/clearShader.glsl?raw";
import combustionShaderSource from "./shaders/combustionShader.glsl?raw";
import curlShaderSource from "./shaders/curlShader.glsl?raw";
import debugFireShaderSource from "./shaders/debugFireShader.glsl?raw";
import debugFloatShaderSource from "./shaders/debugFloatShader.glsl?raw";
import displayShaderSource from "./shaders/displayShader.glsl?raw";
import displayFireShaderSource from "./shaders/displayFireShader.glsl?raw";
import divergenceShaderSource from "./shaders/divergenceShader.glsl?raw";
import pressureIterationShaderSource from "./shaders/pressureIterationShader.glsl?raw";
import projectionShaderSource from "./shaders/projectionShader.glsl?raw";
import rowShaderSource from "./shaders/rowShader.glsl?raw";
import splatShaderSource from "./shaders/splatShader.glsl?raw";
import vorticityConfinementShaderSource from "./shaders/vorticityConfinementShader.glsl?raw";

// Custom Pass for operations that need special handling
class CustomShaderPass extends Pass {
  constructor(shader, uniforms = {}) {
    super();
    
    this.uniforms = uniforms;
    this.material = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader: shader.vertexShader || baseVertexShaderSource,
      fragmentShader: shader.fragmentShader,
      defines: shader.defines || {}
    });
    
    this.fsQuad = new FullScreenQuad(this.material);
  }
  
  render(renderer, writeBuffer, readBuffer) {
    if (this.uniforms.tDiffuse && readBuffer) {
      this.uniforms.tDiffuse.value = readBuffer.texture;
    }
    
    if (this.renderToScreen) {
      renderer.setRenderTarget(null);
    } else if (writeBuffer) {
      renderer.setRenderTarget(writeBuffer);
      if (this.clear) renderer.clear();
    }
    
    this.fsQuad.render(renderer);
  }
  
  setUniforms(uniforms) {
    for (const key in uniforms) {
      if (this.uniforms[key]) {
        this.uniforms[key].value = uniforms[key];
      } else {
        this.uniforms[key] = { value: uniforms[key] };
      }
    }
  }
  
  dispose() {
    this.material.dispose();
    this.fsQuad.dispose();
  }
}

// Helper to create shader objects for three.js passes
function createShader(fragmentShader, vertexShader = baseVertexShaderSource, uniforms = {}) {
  return {
    uniforms: uniforms,
    vertexShader: vertexShader,
    fragmentShader: fragmentShader
  };
}

export function makeFireSimulation() {
  // Create canvas and append to document
  const canvas = document.createElement('canvas');
  canvas.style.position = 'fixed';
  canvas.style.top = '0';
  canvas.style.left = '0';
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.zIndex = '100'; // Put in front of other content
  canvas.style.pointerEvents = 'none'; // Don't interfere with page interactions
  document.body.appendChild(canvas);

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const renderer = new THREE.WebGLRenderer({ 
    canvas: canvas, 
    alpha: true, // Enable alpha for transparency
    premultipliedAlpha: false, // Better control over alpha blending
    depth: false, 
    stencil: false, 
    antialias: false 
  });
  renderer.autoClear = false;
  
  // Enable alpha blending
  renderer.setClearColor(0x000000, 0); // Transparent clear color

  // Get WebGL context from renderer
  const gl = renderer.getContext();
  
  // Helper for WebGL2/Three.js format mapping (simplified)
  const ext = {
    halfFloatTexType: THREE.HalfFloatType,
    floatTexType: THREE.FloatType,
    formatRGBA16F: { internalFormat: gl.RGBA16F, format: THREE.RGBAFormat },
    formatRGBA32F: { internalFormat: gl.RGBA32F, format: THREE.RGBAFormat },
    formatRG16F: { internalFormat: gl.RG16F, format: THREE.RGFormat },
    formatRG32F: { internalFormat: gl.RG32F, format: THREE.RGFormat },
    formatR16F: { internalFormat: gl.R16F, format: THREE.RedFormat },
    formatR32F: { internalFormat: gl.R32F, format: THREE.RedFormat },
    supportLinearFiltering: renderer.extensions.get('OES_texture_float_linear') || renderer.extensions.get('OES_texture_half_float_linear')
  };


  let config = {
    BUOYANCY: 0.1,
    BURN_TEMPERATURE: 1700,
    CONFINEMENT: 15,
    COOLING: 3000,
    DISPLAY_MODE: 0,
    DYE_RESOLUTION: 1024,
    FUEL_DISSIPATION: 0.75,
    DENSITY_DISSIPATION: 0.99,
    NOISE_BLENDING: 0.5,
    NOISE_VOLATILITY: 0.1,
    PRESSURE_DISSIPATION: 0.8,
    PRESSURE_ITERATIONS: 20,
    SIM_RESOLUTION: 512,
    SPLAT_RADIUS: 1.5,
    VELOCITY_DISSIPATION: 0.98,
  };
  let DISPLAY_MODES = ["Normal", "DebugFire", "DebugTemperature", "DebugFuel", "DebugPressure", "DebugDensity", "DebugNoise"];

  let simWidth;
  let simHeight;
  let dyeWidth;
  let dyeHeight;

  let curl;
  let density;
  let divergence;
  let fuel;
  let noise;
  let pressure;
  let temperature;
  let velocity;

  let addNoiseProgram;
  let advectionProgram;
  let buoyancyProgram;
  let clearProgram;
  let combustionProgram;
  let curlProgram;
  let debugFireProgram;
  let debugFloatProgram;
  let displayProgram;
  let displayFireProgram;
  let divergenceProgram;
  let pressureIterationProgram;
  let projectionProgram;
  let rowProgram;
  let splatProgram;
  let vorticityConfinementProgram;
  let imageTexture;
  let imageAspectRatio = 1.0; // Default fallback, will be updated when image loads

  /*
  Render quad to a specified framebuffer `destination`. If null, render to the default framebuffer.
  */

  // Helper function to render a shader pass to a specific target
  function renderPassToTarget(shaderPass, renderTarget, inputTexture = null) {
    if (inputTexture && shaderPass.uniforms && shaderPass.uniforms.tDiffuse) {
      shaderPass.uniforms.tDiffuse.value = inputTexture;
    }
    
    if (renderTarget === null) {
      shaderPass.renderToScreen = true;
    } else {
      shaderPass.renderToScreen = false;
    }
    
    if (shaderPass instanceof CustomShaderPass) {
      shaderPass.render(renderer, renderTarget, null);
    } else {
      // For the local ShaderPass class
      shaderPass.render(renderTarget);
    }
  }

  function getResolution (resolution) {
    let aspectRatio = renderer.domElement.width / renderer.domElement.height;
    if (aspectRatio < 1) {
      aspectRatio = 1.0 / aspectRatio;
    }

    let max = resolution * aspectRatio;
    let min = resolution;

    if (gl.drawingBufferWidth > gl.drawingBufferHeight) {
      return { width: max, height: min };
    } else {
      return { width: min, height: max };
    }
  }

  function initFramebuffers() {
    let simRes = getResolution(config.SIM_RESOLUTION);
    let dyeRes = getResolution(config.DYE_RESOLUTION);

    simWidth = Math.floor(simRes.width);
    simHeight = Math.floor(simRes.height);
    dyeWidth = Math.floor(dyeRes.width);
    dyeHeight = Math.floor(dyeRes.height);

    const texType = ext.halfFloatTexType; // THREE.HalfFloatType
    // Three.js uses its own format constants. We'll map internalFormat and format.
    // The 'format' parameter in WebGLRenderTargetOptions refers to THREE.RGBAFormat etc.
    // The 'type' parameter refers to THREE.UnsignedByteType, THREE.FloatType, THREE.HalfFloatType etc.

    const linearFilter = ext.supportLinearFiltering ? THREE.LinearFilter : THREE.NearestFilter;
    const nearestFilter = THREE.NearestFilter;

    // Mapping WebGL internal formats to Three.js WebGLRenderTargetOptions
    // This is a simplification. For precise control, you might need to delve deeper into Three.js internals
    // or use DataTexture with specific internalFormat if WebGLRenderTarget doesn't expose it directly.

    curl = createFBO(simWidth, simHeight, THREE.RedFormat, texType, nearestFilter);
    density = createDoubleFBO(dyeWidth, dyeHeight, THREE.RGBAFormat, texType, linearFilter);
    divergence = createFBO(simWidth, simHeight, THREE.RedFormat, texType, nearestFilter);
    fuel = createDoubleFBO(simWidth, simHeight, THREE.RedFormat, texType, linearFilter);
    noise = createDoubleFBO(simWidth, simHeight, THREE.RedFormat, texType, linearFilter);
    pressure = createDoubleFBO(simWidth, simHeight, THREE.RedFormat, texType, nearestFilter);
    temperature = createDoubleFBO(simWidth, simHeight, THREE.RedFormat, texType, linearFilter);
    velocity = createDoubleFBO(simWidth, simHeight, THREE.RGFormat, texType, linearFilter);
  }

  function createFBO (w, h, format, type, filter) {
    // const texId = LAST_TEX_ID++; // Three.js handles texture units automatically
    const renderTarget = new THREE.WebGLRenderTarget(w, h, {
      minFilter: filter,
      magFilter: filter,
      format: format,
      type: type,
      wrapS: THREE.ClampToEdgeWrapping,
      wrapT: THREE.ClampToEdgeWrapping,
      depthBuffer: false,
      stencilBuffer: false,
    });

    return {
      texture: renderTarget.texture, // This is the THREE.Texture
      fbo: renderTarget, // This is the THREE.WebGLRenderTarget
      width: w,
      height: h,
    };
  }

  function createDoubleFBO (w, h, format, type, filter) {
    let fbo1 = createFBO(w, h, format, type, filter);
    let fbo2 = createFBO(w, h, format, type, filter);

    return {
      get read () {
        return fbo1;
      },
      get write () {
        return fbo2;
      },
      swap () {
        let temp = fbo1;
        fbo1 = fbo2;
        fbo2 = temp;
      },
    };
  }

  function update () {
    resizeCanvas();
    input();
    step(0.016); // Consider using THREE.Clock for delta time
    render();
    requestAnimationFrame(update);
  }

  const clock = new THREE.Clock(); // For delta time in step()

  function input () {
    // bottom row fire
    rowProgram.setUniforms({
      texelSize: new THREE.Vector2(1.0 / simWidth, 1.0 / simHeight),
      y: config.SIM_RESOLUTION,
      uTarget: fuel.read.texture,
      useMax: true,
      time: clock.getElapsedTime(),
      uImage: imageTexture // Pass the THREE.Texture object
    });
    renderPassToTarget(rowProgram, fuel.write.fbo);
    fuel.swap();

  }

  function resizeCanvas() {
    const container = canvas.parentElement || canvas;
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    
    // Get device pixel ratio for crisp rendering
    const pixelRatio = window.devicePixelRatio || 1;
    
    // Use the actual loaded image aspect ratio
    const containerAspectRatio = containerWidth / containerHeight;
    
    let canvasWidth, canvasHeight;
    let renderWidth, renderHeight;
    
    // Determine canvas size to maintain image aspect ratio while maximizing size
    if (containerAspectRatio > imageAspectRatio) {
      // Container is wider than image - fit to height
      canvasHeight = containerHeight;
      canvasWidth = containerHeight * imageAspectRatio;
    } else {
      // Container is taller than image - fit to width
      canvasWidth = containerWidth;
      canvasHeight = containerWidth / imageAspectRatio;
    }
    
    // Set actual render dimensions with pixel ratio
    renderWidth = Math.floor(canvasWidth * pixelRatio);
    renderHeight = Math.floor(canvasHeight * pixelRatio);
    
    // Update canvas size only if changed
    if (canvas.width !== renderWidth || canvas.height !== renderHeight) {
      canvas.width = renderWidth;
      canvas.height = renderHeight;
      canvas.style.width = canvasWidth + 'px';
      canvas.style.height = canvasHeight + 'px';
      
      renderer.setSize(renderWidth, renderHeight, false);
      camera.updateProjectionMatrix();
      initFramebuffers();
    }
  }

  function render () {
    
    const currentRenderTarget = renderer.getRenderTarget();

    switch(DISPLAY_MODES[config.DISPLAY_MODE]) {
      case "Normal": {
        displayFireProgram.setUniforms({
          uDensity: density.read.texture,
          uTemperature: temperature.read.texture,
          uFuel: fuel.read.texture,
          burnTemperature: config.BURN_TEMPERATURE
        });
        displayFireProgram.renderToScreen = true;
        renderPassToTarget(displayFireProgram, null);
        break;
      }
      case "DebugFire": {
        debugFireProgram.setUniforms({
          uFuel: fuel.read.texture,
          uTemperature: temperature.read.texture,
          temperatureScalar: 0.001,
          fuelScalar: 1.0
        });
        debugFireProgram.renderToScreen = true;
        renderPassToTarget(debugFireProgram, null);
        break;
      }
      case "DebugTemperature": {
        debugFloatProgram.setUniforms({
          uTexture: temperature.read.texture,
          scalar: 0.001
        });
        debugFloatProgram.renderToScreen = true;
        renderPassToTarget(debugFloatProgram, null);
        break;
      }
      case "DebugFuel": {
        debugFloatProgram.setUniforms({
          uTexture: fuel.read.texture,
          scalar: 1.0
        });
        debugFloatProgram.renderToScreen = true;
        renderPassToTarget(debugFloatProgram, null);
        break;
      }
      case "DebugPressure": {
        debugFloatProgram.setUniforms({
          uTexture: pressure.read.texture,
          scalar: 1.0
        });
        debugFloatProgram.renderToScreen = true;
        renderPassToTarget(debugFloatProgram, null);
        break;
      }
      case "DebugNoise": {
        debugFloatProgram.setUniforms({
          uTexture: noise.read.texture,
          scalar: 1.0
        });
        debugFloatProgram.renderToScreen = true;
        renderPassToTarget(debugFloatProgram, null);
        break;
      }
      default: {
        displayProgram.setUniforms({
          uTexture: density.read.texture
        });
        displayProgram.renderToScreen = true;
        renderPassToTarget(displayProgram, null);
        break;
      }
    }
    
    renderer.setRenderTarget(currentRenderTarget);
  }

  // Helper function to check if a number is a power of 2
  function isPowerOf2(value) {
    return (value & (value - 1)) === 0;
  }

  function loadImageTexture() {
    const textureLoader = new THREE.TextureLoader();
    const texture = textureLoader.load(
      './chac-bolay-white-on-black.png',
      (loadedTexture) => {
        // Get the actual image aspect ratio
        imageAspectRatio = loadedTexture.image.width / loadedTexture.image.height;
        
        loadedTexture.minFilter = THREE.LinearFilter; // Or NearestFilter if !isPowerOf2
        loadedTexture.magFilter = THREE.LinearFilter;
        // Check if power of 2 dimensions for mipmapping
        if (isPowerOf2(loadedTexture.image.width) && isPowerOf2(loadedTexture.image.height)) {
          loadedTexture.generateMipmaps = true;
        } else {
          loadedTexture.generateMipmaps = false;
          loadedTexture.wrapS = THREE.ClampToEdgeWrapping;
          loadedTexture.wrapT = THREE.ClampToEdgeWrapping;
        }
        loadedTexture.needsUpdate = true; // Important after setting parameters
        imageTexture = loadedTexture; // Assign to the global scope variable
        
        // Trigger a resize now that we have the correct aspect ratio
        resizeCanvas();
        
        console.log(`Chac-bolay image loaded successfully with Three.js (${loadedTexture.image.width}x${loadedTexture.image.height}, aspect ratio: ${imageAspectRatio})`);
      },
      undefined, // onProgress callback not implemented here
      (err) => {
        console.error('Error loading chac-bolay image with Three.js:', err);
      }
    );
    // Return a placeholder or handle async loading appropriately if needed immediately
    // For now, imageTexture will be updated when the load completes.
    return texture; // Return the initially created (but possibly not yet loaded) texture
  }

  /*
  Update the programs by delta time.
  */
  function step (dt) { // dt from THREE.Clock.getDelta() is usually preferred
    // Combustion step.
    combustionProgram.setUniforms({
      texelSize: new THREE.Vector2(1.0 / simWidth, 1.0 / simHeight),
      uFuel: fuel.read.texture,
      uTemperature: temperature.read.texture,
      uNoise: noise.read.texture,
      noiseBlending: config.NOISE_BLENDING,
      burnTemperature: config.BURN_TEMPERATURE,
      cooling: config.COOLING,
      dt: dt
    });
    renderPassToTarget(combustionProgram, temperature.write.fbo, temperature.read.texture);
    temperature.swap();

    // Advection step for velocity
    advectionProgram.setUniforms({
      texelSize: new THREE.Vector2(1.0 / simWidth, 1.0 / simHeight),
      uVelocity: velocity.read.texture,
      uSource: velocity.read.texture,
      dt: dt,
      dissipation: config.VELOCITY_DISSIPATION
    });
    if (!ext.supportLinearFiltering) {
      advectionProgram.setUniforms({ dyeTexelSize: new THREE.Vector2(1.0 / simWidth, 1.0 / simHeight) });
    }
    renderPassToTarget(advectionProgram, velocity.write.fbo, velocity.read.texture);
    velocity.swap();

    // Vorticity confinement
    curlProgram.setUniforms({
      texelSize: new THREE.Vector2(1.0 / simWidth, 1.0 / simHeight),
      uVelocity: velocity.read.texture,
      uNoise: noise.read.texture,
      blendLevel: config.NOISE_BLENDING
    });
    renderPassToTarget(curlProgram, curl.fbo, velocity.read.texture);

    vorticityConfinementProgram.setUniforms({
      texelSize: new THREE.Vector2(1.0 / simWidth, 1.0 / simHeight),
      uVelocity: velocity.read.texture,
      uCurl: curl.texture,
      confinement: config.CONFINEMENT,
      dt: dt
    });
    renderPassToTarget(vorticityConfinementProgram, velocity.write.fbo, velocity.read.texture);
    velocity.swap();

    // Thermal buoyancy
    buoyancyProgram.setUniforms({
      texelSize: new THREE.Vector2(1.0 / simWidth, 1.0 / simHeight),
      uVelocity: velocity.read.texture,
      uTemperature: temperature.read.texture,
      buoyancy: config.BUOYANCY,
      dt: dt
    });
    renderPassToTarget(buoyancyProgram, velocity.write.fbo, velocity.read.texture);
    velocity.swap();

    // Dissipate pressure
    clearProgram.setUniforms({
      uTexture: pressure.read.texture,
      value: config.PRESSURE_DISSIPATION
    });
    renderPassToTarget(clearProgram, pressure.write.fbo, pressure.read.texture);
    pressure.swap();

    // Projection step
    divergenceProgram.setUniforms({
      texelSize: new THREE.Vector2(1.0 / simWidth, 1.0 / simHeight),
      uVelocity: velocity.read.texture
    });
    renderPassToTarget(divergenceProgram, divergence.fbo, velocity.read.texture);

    // Pressure iteration
    for (let i = 0; i < config.PRESSURE_ITERATIONS; i++) {
      pressureIterationProgram.setUniforms({
        texelSize: new THREE.Vector2(1.0 / simWidth, 1.0 / simHeight),
        uPressure: pressure.read.texture,
        uDivergence: divergence.texture
      });
      renderPassToTarget(pressureIterationProgram, pressure.write.fbo, pressure.read.texture);
      pressure.swap();
    }

    projectionProgram.setUniforms({
      texelSize: new THREE.Vector2(1.0 / simWidth, 1.0 / simHeight),
      uPressure: pressure.read.texture,
      uVelocity: velocity.read.texture
    });
    renderPassToTarget(projectionProgram, velocity.write.fbo, velocity.read.texture);
    velocity.swap();

    // Advect density
    advectionProgram.setUniforms({
      texelSize: new THREE.Vector2(1.0 / dyeWidth, 1.0 / dyeHeight),
      uVelocity: velocity.read.texture,
      uSource: density.read.texture,
      dissipation: config.DENSITY_DISSIPATION,
      dt: dt
    });
    if (!ext.supportLinearFiltering) {
      advectionProgram.setUniforms({ dyeTexelSize: new THREE.Vector2(1.0 / dyeWidth, 1.0 / dyeHeight) });
    }
    renderPassToTarget(advectionProgram, density.write.fbo, density.read.texture);
    density.swap();

    // Advect temperature
    advectionProgram.setUniforms({
      texelSize: new THREE.Vector2(1.0 / simWidth, 1.0 / simHeight),
      uSource: temperature.read.texture,
      dissipation: 1.0
    });
    if (!ext.supportLinearFiltering) {
      advectionProgram.setUniforms({ dyeTexelSize: new THREE.Vector2(1.0 / simWidth, 1.0 / simHeight) });
    }
    renderPassToTarget(advectionProgram, temperature.write.fbo, temperature.read.texture);
    temperature.swap();

    // Advect fuel
    advectionProgram.setUniforms({
      uSource: fuel.read.texture,
      dissipation: config.FUEL_DISSIPATION
    });
    renderPassToTarget(advectionProgram, fuel.write.fbo, fuel.read.texture);
    fuel.swap();

    // Advect noise
    advectionProgram.setUniforms({
      uSource: noise.read.texture,
      dissipation: 1.0
    });
    renderPassToTarget(advectionProgram, noise.write.fbo, noise.read.texture);
    noise.swap();

    // Blend in noise
    addNoiseProgram.setUniforms({
      texelSize: new THREE.Vector2(1.0 / simWidth, 1.0 / simHeight),
      time: clock.getElapsedTime() / 10.0 % 1,
      uTarget: noise.read.texture,
      blendLevel: config.NOISE_VOLATILITY
    });
    renderPassToTarget(addNoiseProgram, noise.write.fbo, noise.read.texture);
    noise.swap();
  }


  window.addEventListener('keydown', (e) => {
    if (e.key === " ") {
      config.DISPLAY_MODE = (config.DISPLAY_MODE + 1) % DISPLAY_MODES.length;
    }
  });

  advectionProgram = new CustomShaderPass(
    createShader(ext.supportLinearFiltering ? advectionShaderSource : advectionManualFilteringShaderSource)
  );
  addNoiseProgram = new CustomShaderPass(
    createShader(addNoiseShaderSource)
  );
  buoyancyProgram = new CustomShaderPass(
    createShader(buoyancyShaderSource)
  );
  clearProgram = new CustomShaderPass(
    createShader(clearShaderSource)
  );
  combustionProgram = new CustomShaderPass(
    createShader(combustionShaderSource)
  );
  curlProgram = new CustomShaderPass(
    createShader(curlShaderSource)
  );
  debugFireProgram = new CustomShaderPass(
    createShader(debugFireShaderSource)
  );
  debugFloatProgram = new CustomShaderPass(
    createShader(debugFloatShaderSource)
  );
  displayProgram = new CustomShaderPass(
    createShader(displayShaderSource)
  );
  displayFireProgram = new CustomShaderPass(
    createShader(displayFireShaderSource)
  );
  divergenceProgram = new CustomShaderPass(
    createShader(divergenceShaderSource)
  );
  pressureIterationProgram = new CustomShaderPass(
    createShader(pressureIterationShaderSource)
  );
  projectionProgram = new CustomShaderPass(
    createShader(projectionShaderSource)
  );
  rowProgram = new CustomShaderPass(
    createShader(rowShaderSource)
  );
  splatProgram = new CustomShaderPass(
    createShader(splatShaderSource)
  );
  vorticityConfinementProgram = new CustomShaderPass(
    createShader(vorticityConfinementShaderSource)
  );


  initFramebuffers();
  
  // Load the image texture using Three.js TextureLoader
  loadImageTexture(); // imageTexture will be assigned asynchronously

  // Initialize the noise channel.
  addNoiseProgram.setUniforms({
    texelSize: new THREE.Vector2(1.0 / simWidth, 1.0 / simHeight),
    time: clock.getElapsedTime() / 100.0 % 1,
    uTarget: noise.read.texture,
    blendLevel: 1.0
  });
  renderPassToTarget(addNoiseProgram, noise.write.fbo, noise.read.texture);
  noise.swap();

  update(); // Start the animation loop
  
  // Return the canvas in case the caller needs access to it
  return canvas;
}
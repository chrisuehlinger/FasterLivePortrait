/*
Fill a texel row below the y-coordinate with noise in the target texture.
*/

precision highp float;
precision mediump sampler2D;

varying vec2 vUv;
uniform sampler2D uTarget; // target texture to create the row
uniform sampler2D uImage;  // Added the image texture uniform
uniform float y; // y-coordinate of the row (in grid coordinates).
uniform bool useMax; // if TRUE, output is max rather than additive.
uniform vec2 texelSize; // simulation grid width.
uniform float time;  // Added time uniform

// Simple 2D noise function (pseudo-random)
float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Smooth interpolation
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.y * u.x;
}

void main () {
  vec2 uv = gl_FragCoord.xy * texelSize;
  vec3 base = texture2D(uTarget, vUv).xyz;
  
  // Get color from the image - can be used for custom fuel patterns
  vec2 imageUV = vUv;
  
  // Turbulent displacement
  float turbulenceStrength = 0.05; // Adjust for more or less turbulence
  float noiseValX = noise(imageUV * 10.0 + time * 0.1); // Scale UV and add time for animation
  float noiseValY = noise(imageUV * 10.0 + time * 0.1 + vec2(5.2, 1.3)); // Offset for different noise pattern

  vec2 displacement = vec2(noiseValX - 0.5, noiseValY - 0.5) * turbulenceStrength; // Center noise and apply strength
  imageUV += displacement;

  // Add time-based offset to create animation effect (existing)
  imageUV.x = imageUV.x + sin(time * 0.2) * 0.02; // Reduced strength of original sine wave movement
  vec3 imgColor = texture2D(uImage, imageUV).xyz;
  
  // Add fuel at row y
  if (gl_FragCoord.y < y) {
    if (useMax) {
      gl_FragColor = vec4(max(base, imgColor), 1.0);
    } else {
      gl_FragColor = vec4(base + imgColor, 1.0);
    }
  } else {
    gl_FragColor = vec4(base, 1.0);
  }
}

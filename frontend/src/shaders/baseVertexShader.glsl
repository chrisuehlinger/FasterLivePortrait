precision highp float;
precision mediump sampler2D;

uniform vec2 texelSize; // To be supplied from JS: vec2(1.0/textureWidth, 1.0/textureHeight)

varying vec2 vUv;
varying vec2 vL;
varying vec2 vR;
varying vec2 vT;
varying vec2 vB;

void main() {
    vUv = uv; // Pass the UV coordinates to the fragment shader

    // Calculate UV coordinates for neighboring texels
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y); // In WebGL/Three.js UVs, +Y is typically up
    vB = vUv - vec2(0.0, texelSize.y); // -Y is typically down

    // For a PlaneGeometry(2,2) and an OrthographicCamera covering -1 to 1 in X and Y,
    // position.xy is already in clip space.
    gl_Position = vec4(position.xy, 0.0, 1.0);
}

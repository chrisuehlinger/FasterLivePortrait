precision highp float;
precision mediump sampler2D;

varying vec2 vUv;
uniform sampler2D uDensity;
uniform sampler2D uTemperature;
uniform sampler2D uFuel;
uniform float burnTemperature;

void main () {
  vec3 c = texture2D(uDensity, vUv).rgb;
  float T = texture2D(uTemperature, vUv).r;
  float F = texture2D(uFuel, vUv).r;

  float heat = T / burnTemperature;
  float fuel = F;
  
  // Color mapping for fire
  vec3 fireColor = vec3(0.0);
  
  // Base fire color from heat
  if (heat > 0.0) {
    fireColor = mix(vec3(0.0), vec3(1.0, 0.0, 0.0), clamp(heat * 2.0, 0.0, 1.0)); // Black to red
    fireColor = mix(fireColor, vec3(1.0, 0.5, 0.0), clamp((heat - 0.5) * 2.0, 0.0, 1.0)); // Red to orange
    fireColor = mix(fireColor, vec3(1.0, 1.0, 0.0), clamp((heat - 0.75) * 4.0, 0.0, 1.0)); // Orange to yellow
    fireColor = mix(fireColor, vec3(1.0, 1.0, 1.0), clamp((heat - 0.9) * 10.0, 0.0, 1.0)); // Yellow to white
  }
  
  // Add density contribution
  fireColor += c;
  
  // Calculate alpha based on fire intensity
  float fireIntensity = max(max(fireColor.r, fireColor.g), fireColor.b);
  float alpha = clamp(fireIntensity, 0.0, 1.0);
  
  gl_FragColor = vec4(fireColor, alpha);
}

// vite.config.js
import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: 'src',
  base: './',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      input: {
        actor: resolve(__dirname, 'src/actor.html'),
        director: resolve(__dirname, 'src/director.html'),
        viewer: resolve(__dirname, 'src/viewer.html'),
      }
    }
  }
});
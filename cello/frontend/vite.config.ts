import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 4000, // Development server port
  },
  build: {
    outDir: 'dist', // Production build directory
    emptyOutDir: true, // Clear the output directory before building
  },
});

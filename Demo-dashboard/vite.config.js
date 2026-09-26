import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Registers the React plugin (automatic JSX runtime + fast refresh) so JSX files
// build without an explicit `import React`. Output goes to dist/ (Vercel Vite preset).
export default defineConfig({
  plugins: [react()],
});

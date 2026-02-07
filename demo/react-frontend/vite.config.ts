import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// dev proxy: routes starting with /api -> http://127.0.0.1:8000
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  server: mode === 'development' ? {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  } : undefined,
}))
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles/index.css'
import App from './App.tsx'

const THEME_KEY = 'theme';
const saved = localStorage.getItem(THEME_KEY); // 'light' | 'dark' | null
if (saved) document.body.classList.toggle('dark-mode', saved === 'dark');

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

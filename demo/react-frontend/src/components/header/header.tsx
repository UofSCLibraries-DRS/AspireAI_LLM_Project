import { useState } from 'react';
import { Sun, Moon } from 'lucide-react';
import './header.css'

function Header() {
  const THEME_KEY = 'theme';
  const saved = (localStorage.getItem(THEME_KEY) as 'light' | 'dark' | null) ?? 'light';
  const [isDark, setIsDark] = useState(saved === 'dark');

  const toggleSidebar = () => {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.querySelector('.sidebar-overlay');
    if (sidebar) sidebar.classList.toggle('open');
    if (overlay) overlay.classList.toggle('show');
  }

  const toggleTheme = () => {
    const next = !isDark;
    document.body.classList.toggle('dark-mode', next);
    localStorage.setItem(THEME_KEY, next ? 'dark' : 'light');
    setIsDark(next);
  }

  return (
    <div id="header">
      <div style={{ display: "flex", alignItems: "center", gap: "15px" }}>
        <button id="menu-toggle" onClick={toggleSidebar}>☰</button>
        <h1>AspireAI LLM Project</h1>
      </div>
      <button id="themeToggle" onClick={toggleTheme} aria-label="Toggle theme">
        {isDark ? <Moon size={18} /> : <Sun size={18} />}
      </button>
    </div>
  );
}

export default Header;
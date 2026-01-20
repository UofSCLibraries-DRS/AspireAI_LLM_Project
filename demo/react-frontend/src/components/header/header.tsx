// import ThemeToggle from "../buttons/theme-toggle"
import './header.css'

function Header() {
  const toggleSidebar = () => {
      const sidebar = document.getElementById('sidebar');
      const overlay = document.querySelector('.sidebar-overlay');
      if (sidebar) {
        sidebar.classList.toggle('open');
      }
      if (overlay) {
        overlay.classList.toggle('show');
      }
    }

  const toggleTheme = () => {
      document.body.classList.toggle('dark-mode');
      const themeBtn = document.getElementById('themeToggle');
      if (themeBtn){
        themeBtn.textContent = document.body.classList.contains('dark-mode') ? '🌙' : '☀️';
        localStorage.setItem('theme', document.body.classList.contains('dark-mode') ? 'dark' : 'light');
      } 
    }

  return (
    <div id="header">
      <div style={{ display: "flex", alignItems: "center", gap: "15px" }}>
        <button id="menu-toggle" onClick={toggleSidebar}>☰</button>
        <h1>
          AspireAI LLM Project
        </h1>
      </div>
      {/* <ThemeToggle/> */}
      <button id="themeToggle" onClick={toggleTheme}>☀️</button>
    </div>
  );
}

export default Header;
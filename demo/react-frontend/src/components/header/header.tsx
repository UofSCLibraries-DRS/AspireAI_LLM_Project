// import ThemeToggle from "../buttons/theme-toggle"
import './styles.modal.css'

function Header() {
  const toggleSidebar = () => {
    // Add your sidebar toggle logic here
  };

  return (
    <div id="header">
      <div style={{ display: "flex", alignItems: "center", gap: "15px" }}>
        <button className="menu-toggle" onClick={toggleSidebar}>☰</button>
        <h1>
          AspireAI LLM Project
        </h1>
      </div>
      {/* <ThemeToggle/> */}
      <button id="themeToggle">☀️</button>
    </div>
  );
}

export default Header;
export function openModal(modalId: string) {
  const el = document.getElementById(modalId);
  if (el) el.style.display = 'flex';
}

export function closeModal(modalId: string) {
  const el = document.getElementById(modalId);
  if (el) el.style.display = 'none';
}

export const toggleSidebar = () => {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.querySelector('.sidebar-overlay');
  if (sidebar) {
    sidebar.classList.toggle('open');
  }
  if (overlay) {
    overlay.classList.toggle('show');
  }
}
export function openModal(modalId: string) {
  const el = document.getElementById(modalId);
  if (el) el.style.display = 'flex';
}

export function closeModal(modalId: string) {
  const el = document.getElementById(modalId);
  if (el) el.style.display = 'none';
}

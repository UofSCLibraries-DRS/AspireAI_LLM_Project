import { useEffect } from 'react';
import type { ReactNode } from 'react';
import './modals.css'

interface GenericModalProps {
  show: boolean;
  onClose: () => void;
  title?: string;
  children: ReactNode;
}

function GenericModal({ show, onClose, title, children }: GenericModalProps) {
  // Esc to exit
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && show) {
        onClose();
      }
    };
    if (show) document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [show, onClose]);

  if (!show) return null;

  return (
    <div className="modal" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="close-modal" onClick={onClose}>×</button>
        {title && <h3>{title}</h3>}
        {children}
      </div>
    </div>
  );
}

export default GenericModal;
import GenericModal from './generic-modal';
import './modals.css'

function GaicoIntroModal({ show, onClose }: { show: boolean; onClose: () => void }) {
  return(
    <GenericModal show={show} onClose={onClose} title="Compare AI Responses with AI">
      <p>prompt aakmkaknanknvknkvneknvkneve vekvnkenkveknvnkeknvnkevknnkevnkenkvnkevnkkvnenkvn</p>
      <p>response aakmkaknanknvknkvneknvkneve vekvnkenkveknvnkeknvnkevknnkevnkenkvnkevnkkvnenkvn</p>
      <p>GenAI Results Comparator, GAICo, can compare MODEL's response to other models.</p>
      <button>Compare</button>
    </GenericModal>
  );
}

export default GaicoIntroModal;
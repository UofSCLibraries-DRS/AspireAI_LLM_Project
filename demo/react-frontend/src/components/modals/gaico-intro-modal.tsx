import GenericModal from './generic-modal';
import './modals.css'

import type { GaicoRequest } from '../../constants/types'

interface GaicoIntroModalProps {
  show: boolean;
  onClose: () => void;
  gaicoReq: GaicoRequest;
}

function GaicoIntroModal({ show, onClose, gaicoReq }: GaicoIntroModalProps) {
  return(
    <GenericModal show={show} onClose={onClose} title="Compare AI Responses with AI">
      <p>prompt - {gaicoReq.prompt}</p>
      <p>response - {gaicoReq.chatbotResponse}</p>
      <p>GenAI Results Comparator, GAICo, can compare {gaicoReq.modelName}'s ({gaicoReq.apiValue}) response to other models.</p>
      <button>Compare</button>
    </GenericModal>
  );
}

export default GaicoIntroModal;
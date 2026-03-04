import { useNavigate } from 'react-router-dom';
import GenericModal from './generic-modal';
import './modals.css'

import type { GaicoRequest } from '../../constants/types'

interface GaicoIntroModalProps {
  show: boolean;
  onClose: () => void;
  gaicoReq: GaicoRequest;
}

function GaicoIntroModal({ show, onClose, gaicoReq }: GaicoIntroModalProps) {
  const navigate = useNavigate();

  const handleCompare = () => {
    // Navigate to GAICo page with state containing the comparison data
    navigate('/gaico', { 
      state: { 
        gaicoRequest: gaicoReq 
      } 
    });
    onClose();
  };

  return(
    <GenericModal show={show} onClose={onClose} title="Compare AI Responses with GAICo">
      <p><strong>Prompt:</strong> {gaicoReq.prompt}</p>
      <p><strong>Response from {gaicoReq.modelName}:</strong> {gaicoReq.chatbotResponse}</p>
      <p>GenAI Results Comparator, GAICo, can compare {gaicoReq.modelName}'s ({gaicoReq.apiValue}) response to other models.</p>
      <button onClick={handleCompare}>Compare with GAICo</button>
    </GenericModal>
  );
}

export default GaicoIntroModal;
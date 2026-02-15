import { GitCompareArrows } from 'lucide-react';
import GaicoIntroModal from '../../components/modals/gaico-intro-modal'
import { useState } from 'react';
import type { GaicoRequest } from '../../constants/types'

interface ChatMessageProps {
  text: string;
  type: 'user' | 'bot';
  info?: string;
  gaicoObject: GaicoRequest;
}

function ChatMessage({ text, type, info, gaicoObject }: ChatMessageProps) {
  const [isModalOpen, setModalOpen] = useState(false);

  return (
    <div className={`message-wrapper ${type}`}>
      <div className={`message ${type}`}>{text}</div>
      <div className="message-below">
        {info && <div className="message-info">{info}</div>}
        {type === "bot" && (
          <div className="clickable-icon">
            <div className="compare-button" onClick={() => setModalOpen(true)}>
              <div>GAICo</div>
              <GitCompareArrows onClick={() => setModalOpen(true)} />
            </div>
          </div>
        )}
      </div>

      <GaicoIntroModal show={isModalOpen} onClose={() => setModalOpen(false)} gaicoReq={gaicoObject} />
    </div>
  );
}

export default ChatMessage;
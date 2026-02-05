import { GitCompareArrows } from 'lucide-react';

interface ChatMessageProps {
  text: string;
  type: 'user' | 'bot';
  info?: string;
}

function ChatMessage({ text, type, info }: ChatMessageProps) {
  const handleCompare = () => {

  }

  return (
    <div className={`message-wrapper ${type}`}>
      <div className={`message ${type}`}>{text}</div>
      <div className="message-below">
        {info && <div className="message-info">{info}</div>}
        {type === "bot" && (
          <div className="clickable-icon">
          <div className="compare-button" onClick={handleCompare}>
            <div>GAICo</div>
            <GitCompareArrows onClick={handleCompare}/>
          </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ChatMessage;
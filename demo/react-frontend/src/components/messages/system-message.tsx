interface SystemMessageProps {
  text: string;
}

function SystemMessage({ text }: SystemMessageProps) {
  return <div className="system-message">{text}</div>;
}

export default SystemMessage;
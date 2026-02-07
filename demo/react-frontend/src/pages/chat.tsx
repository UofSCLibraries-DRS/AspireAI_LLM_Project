import Header from '../components/header/header'
import Sidebar from '../components/sidebar/sidebar'
import { useEffect, useRef, useState } from 'react';
import { toggleSidebar } from '../services/modals'
import ModelSelector from '../components/model-selector/model-selector'
import ChatMessage from "../components/messages/chat-message"
import SystemMessage from "../components/messages/system-message"
import { GENERATE_API } from "../api";
import { useModels } from '../hooks/useModels';

import '../styles/index.css'

interface Message {
  id: string;
  text: string;
  type: 'user' | 'bot' | 'system';
  info?: string;
}

function Chat() {
    const [selectedModel, setSelectedModel] = useState('M8')
    const [messages, setMessages] = useState<Message[]>([])
    const { models } = useModels();

    const refs = {
      chatLog: useRef<HTMLDivElement | null>(null),
      userInput: useRef<HTMLTextAreaElement | null>(null),
      sendBtn: useRef<HTMLButtonElement | null>(null),
    };

    useEffect(() => {
      const handleOutsideModalClick = (e: MouseEvent) => {
          const target = e.target as HTMLElement | null;
          if (target?.classList.contains('modal')) {
            target.style.display = 'none';
          }
        };
        window.addEventListener('click', handleOutsideModalClick);
        return () => window.removeEventListener('click', handleOutsideModalClick);
    }, []);

    // Auto-scroll to bottom when messages change
    useEffect(() => {
      const chatLog = refs.chatLog.current;
      if (chatLog) {
        chatLog.scrollTop = chatLog.scrollHeight;
      }
    }, [messages]);

    async function sendMessage() {
      const userInput = refs.userInput.current;
      const sendBtn = refs.sendBtn.current

      if (!userInput || !sendBtn ) { return; }

      const message = userInput.value.trim();
      if (!message) { return; }

      const selectedModelValue = selectedModel;
      const modelData = models.find(m => m.apiValue === selectedModelValue);
      const modelDisplayName = modelData ? modelData.name : selectedModelValue;

      // Add user message to state 
      setMessages(prev => [...prev, { 
        id: Date.now().toString(), 
        text: message, 
        type: 'user' 
      }]);
      
      userInput.value = '';
      userInput.style.height = 'auto';

      sendBtn.disabled = true;
      sendBtn.textContent = 'Sending...';

      try {
        const response = await fetch(GENERATE_API, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            prompt: message,
            model: selectedModelValue,
            max_new_tokens: 128
          })
        });

        if (!response.ok) {
          throw new Error(`API error ${response.status}`);
        }

        const data = await response.json();
        const botText = data?.text || "No response received";

        setMessages(prev => [...prev, { 
          id: Date.now().toString(), 
          text: botText, 
          type: 'bot', 
          info: modelDisplayName 
        }]);

        console.log("Curr model: ", selectedModel)

      } catch (error) {
        console.error('Error:', error);
        setMessages(prev => [...prev, { 
          id: Date.now().toString(), 
          text: 'The response engine encountered an error. Please try again.', 
          type: 'system' 
        }]);
      } finally {
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send';
      }
    }

    return (
        <div>
            <Header/>

            <div id="mainContainer">
  
                <Sidebar/>

                <div id="chatContainer">
                    
                    <div id="chatLog" ref={refs.chatLog}>
                        {messages.length === 0 ? (
                          <div id="welcomeScreen">
                              <h2>Welcome to AspireAI LLM Project</h2>
                              <p>Start a conversation by typing your question below. Select a model from the dropdown to begin.</p>
                          </div>
                        ) : (
                          messages.map(msg => 
                            msg.type === 'system' ? (
                              <SystemMessage key={msg.id} text={msg.text} />
                            ) : (
                              <ChatMessage key={msg.id} text={msg.text} type={msg.type} info={msg.info} />
                            )
                          )
                        )}
                    </div>

                    <div id="inputArea">
                        <div className="input-controls">
                            <ModelSelector 
                                value={selectedModel} 
                                onChange={setSelectedModel} 
                            />
                            <div className="button-row">
                                <textarea
                                    id="userInput"
                                    ref={refs.userInput}
                                    placeholder="Type your message here..."
                                    rows={1}
                                    onInput={(e) => {
                                        const el = e.currentTarget;
                                        el.style.height = 'auto';
                                        el.style.height = `${el.scrollHeight}px`;
                                    }}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter' && !e.shiftKey) {
                                        e.preventDefault();
                                        sendMessage();
                                        }
                                    }}
                                ></textarea>
                                <button id="sendBtn" ref={refs.sendBtn} onClick={sendMessage}>Send</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="sidebar-overlay" onClick={toggleSidebar}/> 
  
        </div>
    );
}

export default Chat;
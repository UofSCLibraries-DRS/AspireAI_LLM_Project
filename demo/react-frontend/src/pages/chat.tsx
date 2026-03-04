import Header from '../components/header/header'
import Sidebar from '../components/sidebar/sidebar'
import { useEffect, useRef, useState } from 'react';
import { toggleSidebar } from '../services/modals'
import ModelSelector from '../components/model-selector/model-selector'
import ChatMessage from "../components/messages/chat-message"
import SystemMessage from "../components/messages/system-message"
import { generateResponse } from "../services/apiClient";
import { useModels } from '../hooks/useModels';
import type { GaicoRequest } from '../constants/types'

import '../styles/index.css'

interface Message {
  id: string;
  text: string;
  type: 'user' | 'bot' | 'system';
  info?: string;
  prompt?: string;    
  response?: string;
  modelApiValue?: string;
}

function Chat() {
    const [selectedModel, setSelectedModel] = useState('M9')
    const [messages, setMessages] = useState<Message[]>([])
    const { models } = useModels();

    const refs = {
      chatLog: useRef<HTMLDivElement | null>(null),
      userInput: useRef<HTMLTextAreaElement | null>(null),
      sendBtn: useRef<HTMLButtonElement | null>(null),
    };

    // handle all modals + behavior of sidebar modal 
    useEffect(() => {

        const handleResize = () => {
          const sidebar = document.getElementById('sidebar');
          const overlay = document.querySelector('.sidebar-overlay');
          const chatLog = refs.chatLog.current;

          // Preserve scroll positions
          const pageScrollY = window.scrollY;
          const chatScrollTop = chatLog?.scrollTop ?? 0;

          // Ensure screen does not stay gray upon resize (sidebar behavior)
          if (window.innerWidth > 768) {
            if (sidebar) sidebar.classList.remove('open');
            if (overlay) overlay.classList.remove('show');
          }

          requestAnimationFrame(() => {
            window.scrollTo(0, pageScrollY);
            if (chatLog) chatLog.scrollTop = chatScrollTop;
          });
        };

        window.addEventListener('resize', handleResize);
        handleResize();

        return () => {
          window.removeEventListener('resize', handleResize);
        };
    }, []);

    // Auto-scroll to bottom when messages change
    useEffect(() => {
      const chatLog = refs.chatLog.current;
      if (chatLog) {
        chatLog.scrollTop = chatLog.scrollHeight;
      }
    }, [messages]);

    // Create id (time + uuid)
    const createId = () => {
      return (typeof crypto !== 'undefined' && 'randomUUID' in crypto)
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    }

    // Function to send message
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
        const data = await generateResponse(message, selectedModelValue);
        const botText = data?.text || "No response received";

        setMessages(prev => [...prev, { 
          id: createId(), 
          text: botText, 
          type: 'bot', 
          info: modelDisplayName,
          prompt: message,
          response: botText,
          modelApiValue: selectedModelValue
        }]);

        console.log("Curr model: ", selectedModel)

      } catch (error) {
        console.error('Error:', error);
        setMessages(prev => [...prev, { 
          id: createId(), 
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
                              <ChatMessage 
                                key={msg.id} text={msg.text} type={msg.type} info={msg.info} 
                                // modelName: display name for model
                                // apiValue: backend model ID (e.g., "M8")
                                // prompt: user input text
                                // chatbotResponse: model response text
                                gaicoObject={{
                                  modelName: msg.info || '',
                                  apiValue: msg.modelApiValue || selectedModel,
                                  prompt: msg.prompt || '',
                                  chatbotResponse: msg.response || msg.text
                                } as GaicoRequest} 
                              />
                            )
                          )
                        )}
                    </div>

                    <div id="inputArea">
                        <div className="input-controls">
                            <ModelSelector 
                                value={selectedModel} 
                                onChange={setSelectedModel} 
                                models={models}
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
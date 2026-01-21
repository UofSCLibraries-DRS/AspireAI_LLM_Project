import Header from '../components/header/header'
import Sidebar from '../components/sidebar/sidebar'
import { useEffect, useRef } from 'react';
import { toggleSidebar } from '../services/modals'

import '../styles/index.css'

// Model Configuration
const AVAILABLE_MODELS = [
    { name: 'Gemma-3-270m, Fine-tuned', apiValue: 'M8' },
    { name: 'SafeChat', apiValue: 'SC' },
];

// Generation API endpoint
const GENERATE_API = "https://54.162.34.113/api/generate";

function MainChat() {

    // Group DOM refs
    const refs = {
        chatLog: useRef<HTMLDivElement | null>(null),
        welcome: useRef<HTMLDivElement | null>(null),
        userInput: useRef<HTMLTextAreaElement | null>(null),
        sendBtn: useRef<HTMLButtonElement | null>(null),
        modelSelector: useRef<HTMLSelectElement | null>(null),
    };

    // Handle outside clicks for modal removal
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
    
    // Add message to chat
    const addMessage = (text: string, type: string, info = '') => {
        const chatLog = refs.chatLog.current;
        const welcome = refs.welcome.current;
        if (!chatLog) return;

        if (welcome && welcome.parentNode) {
        welcome.remove();
        }

        const wrapper = document.createElement('div');
        wrapper.className = `message-wrapper ${type}`;

        const message = document.createElement('div');
        message.className = `message ${type}`;
        message.textContent = text;
        wrapper.appendChild(message);

        if (info) {
        const infoDiv = document.createElement('div');
        infoDiv.className = 'message-info';
        infoDiv.textContent = info;
        wrapper.appendChild(infoDiv);
        }

        chatLog.appendChild(wrapper);
        chatLog.scrollTop = chatLog.scrollHeight;
    };

    // 
    function addSystemMessage(text: string) {
      const chatLog = refs.chatLog.current;
      const welcome = refs.welcome.current;
      if (!chatLog) return;

      if (welcome && welcome.parentNode) {
        welcome.remove();
      }

      const message = document.createElement('div');
      message.className = 'system-message';
      message.textContent = text;
      chatLog.appendChild(message);
      chatLog.scrollTop = chatLog.scrollHeight;
    }


    async function sendMessage() {
      const userInput = refs.userInput.current;
      const modelSelector = refs.modelSelector.current
      const sendBtn = refs.sendBtn.current

      if (!userInput || !modelSelector || !sendBtn ) { return; }

      const message = userInput.value.trim();
      if (!message) { return; }

      const selectedModelValue = modelSelector.value;
      const selectedModel = AVAILABLE_MODELS.find(m => m.apiValue === selectedModelValue);
      const modelDisplayName = selectedModel ? selectedModel.name : selectedModelValue;

      addMessage(message, 'user');
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

        addMessage(botText, 'bot', modelDisplayName);

      } catch (error) {
        console.error('Error:', error);
        addSystemMessage('The response engine encountered an error. Please try again.');
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

                {/* <!-- Chat Container --> */}
                <div id="chatContainer">
                    
                    <div id="chatLog" ref={refs.chatLog}>
                        <div id="welcomeScreen" ref={refs.welcome}>
                            <h2>Welcome to AspireAI LLM Project</h2>
                            <p>Start a conversation by typing your question below. Select a model from the dropdown to begin.</p>
                        </div>
                    </div>

                    {/* <!-- Input Area --> */}
                    <div id="inputArea">
                        <div className="input-controls">
                            <select id="modelSelector" ref={refs.modelSelector}>
                                {/* load models */}
                                {AVAILABLE_MODELS.map(model => (
                                    <option key={model.apiValue} value={model.apiValue}>
                                    {model.name}
                                    </option>
                                ))}
                            </select>
                            <div className="button-row">
                                {/* handle user input (text-box & send) */}
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
                                {/* TODO: add survey submission */}
                                <button id="surveyBtn" onClick={sendMessage}>View Survey</button>
                            </div>
                        </div>
                    </div>
                </div> {/* chatContainer */}
            </div> {/* mainContainer */}

            {/* menu handling for sidebar on smaller windows / mobile */}
            <div className="sidebar-overlay" onClick={toggleSidebar}/> 
  
        </div>
    );
}

export default MainChat;

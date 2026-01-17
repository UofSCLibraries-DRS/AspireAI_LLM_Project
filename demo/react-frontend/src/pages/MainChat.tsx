import Header from '../components/header/header'
import Sidebar from '../components/sidebar/sidebar'
import '../styles/index.css'

function MainChat() {
    // Model Configuration
    const AVAILABLE_MODELS = [
      { name: 'Gemma-3-270m, Fine-tuned', apiValue: 'M8' },
      { name: 'SafeChat', apiValue: 'SC' },
    ];

    const chatLog = document.getElementById('chatLog');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const modelSelector = document.getElementById('modelSelector');

    // Initialize model selector
    function initializeModelSelector() {
      AVAILABLE_MODELS.forEach(model => {
        const option = document.createElement('option');
        option.value = model.apiValue;
        option.textContent = model.name;
        if (modelSelector != null) {
            modelSelector.appendChild(option);
        }
      });
    }

    return (
        <div>
            <Header/>

            <div id="mainContainer">
                <Sidebar/>

                {/* <!-- Chat Container --> */}
                <div id="chatContainer">
                    
                    <div id="chatLog">
                        <div id="welcomeScreen">
                            <h2>Welcome to AspireAI LLM Project</h2>
                            <p>Start a conversation by typing your question below. Select a model from the dropdown to begin.</p>
                        </div>
                    </div>

                    {/* <!-- Input Area --> */}
                    <div id="inputArea">
                        <div className="input-controls">
                            <select id="modelSelector">
                                {/* <!-- Options will be populated by JavaScript --> */}
                            </select>
                            {/* onclick */}
                            <div className="button-row">
                                <textarea id="userInput" placeholder="Type your message here..." rows="1"></textarea>
                                <button id="sendBtn">Send</button>
                            </div>
                        </div>
                        {/* onclick */}
                        <button id="surveyBtn">View Survey</button>
                    </div>
                </div>
            </div>

            {/* onClick */}
            {/* mobile overlay */}
            <div className="sidebar-overlay"/> 
  
        </div>
    );
}

export default MainChat;

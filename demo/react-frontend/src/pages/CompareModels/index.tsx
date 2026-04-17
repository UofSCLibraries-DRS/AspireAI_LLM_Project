import Header from '../../components/header/header'
import { useState } from 'react'
import ModelSelector from '../../components/model-selector/model-selector'
import '../../styles/index.css'

function StartCompare() {
    const [page, setPage] = useState("start");

    const [selectedModel_1, setSelectedModel_1] = useState('M8')
    const [selectedModel_2, setSelectedModel_2] = useState('SC')

    const handleSendMessage = () => {
        // Send selectedModel to backend (or wherever)
        console.log('Selected model:', selectedModel_1)
    }


    return (
        <div>
            <Header/>

            <div id="centerHorizContainer"> 
                <h2>Select Models to Compare</h2>
            </div>

            <div id="centerHorizContainer"> 
                <ModelSelector 
                    value={selectedModel_1} 
                    onChange={setSelectedModel_1} 
                />

                <ModelSelector 
                    value={selectedModel_2} 
                    onChange={setSelectedModel_2} 
                />

                <button onClick={console.log("This does nothing currently")}>+</button>
            </div>

            <div id="centerHorizContainer"> 
                <button onClick={handleSendMessage}>Continue</button>
            </div>
        </div>
    );
}

export default StartCompare 
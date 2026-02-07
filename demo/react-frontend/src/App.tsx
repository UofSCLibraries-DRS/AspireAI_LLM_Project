import {
  BrowserRouter as Router,
  Routes, 
  Route
} from "react-router"

import MainChat from "./pages/chat"
// import CompareModels from './pages/CompareModels'  
// import PromptSelect from './pages/CompareModels/prompt-select' 
// import Results from './pages/CompareModels/results'  
import './styles/index.css'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainChat />} />
        {/* <Route path="/compare" element={<CompareModels /> } /> */}
      </Routes>
    </Router>
  );
}

export default App

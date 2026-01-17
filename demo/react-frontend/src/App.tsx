import {
  BrowserRouter as Router,
  Routes, 
  Route
} from "react-router"

import MainChat from "./pages/MainChat"
import CompareChat from "./pages/CompareChat"
import CompareResults from "./pages/MainChat"
import './styles/index.css'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainChat />} />
        <Route path="/compare-chat" element={<CompareChat />} />
        <Route path="/compare-results" element={<CompareResults />} />
      </Routes>
    </Router>
  );
}

export default App

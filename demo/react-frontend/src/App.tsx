import {
  BrowserRouter as Router,
  Routes, 
  Route
} from "react-router"

import MainChat from "./pages/chat"
import Gaico from "./pages/gaico"
import './styles/index.css'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainChat />} />
        <Route path="/gaico" element={<Gaico /> } />
      </Routes>
    </Router>
  );
}

export default App

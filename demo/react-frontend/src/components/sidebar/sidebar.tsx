import { useState } from "react";
import PolicyModal from "../modals/policy-modal"
import './styles.modal.css'

function Sidebar() {
    // const [isOpen, setOpen] = useState(false);
    const [isPolicyOpen, setPolicyOpen] = useState(false);

    return (
    <div id="sidebar">
      <h2>About the Chatbot</h2>
      <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore
        magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
        consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla
        pariatur.</p>

        <button className="sidebar-button" onClick={() => setPolicyOpen(true)}>
        View Privacy Policy 
      </button>

      <PolicyModal show={isPolicyOpen} onClose={() => setPolicyOpen(false)} />
    
      {/* temporary link to collections */}
      <button 
        className="sidebar-button" 
        onClick={() => window.open('https://sc.edu/about/offices_and_divisions/university_libraries/browse/digital_collections/index.php', '_blank', 'noopener,noreferrer')}
      >
        Learn More About This Project
      </button>

      {/* TO-DO: Update sample questions to better represent where our bot preforms decently */}

      <h3>Sample Questions</h3>
      <p>The system can answer various types of questions. Here are some suggestions to get started:</p>
      <ul>
        <li>Who was John H. McCray?</li>
        <li>What was John H. McCray's position in the Progressive Democratic Party?</li>
      </ul>
      <h3>Limitations</h3>
      <p>The models currently available lack:</p>
      <ul>
        <li>Use of context from previous questions and responses for follow-up questions</li>
        <li>Access to information outside of the John H. McCray digital collection</li>
        <li>Citation of sources</li>
      </ul>

      {/* <h3>In-Scope Questions:</h3>
      <ul>
        <li>General knowledge inquiries</li>
        <li>Technical explanations</li>
        <li>Creative writing requests</li>
      </ul>

      <h3>Do-Not-Answer Questions:</h3>
      <ul>
        <li>Requests for harmful content</li>
        <li>Personal data about individuals</li>
        <li>Medical or legal advice</li>
      </ul>  */}

    </div>

    );
}

export default Sidebar;
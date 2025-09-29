// import { saveMessageFeedback, saveSurveyFeedback } from "./firebase.js";

const chatLog = document.getElementById("chatLog");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const survey = document.getElementById("survey");
const ratingInput = document.getElementById("rating");
const commentsInput = document.getElementById("comments");
const submitFeedbackBtn = document.getElementById("submitFeedbackBtn");
const senderID = "user_" + Math.floor(Math.random() * 100000);
let lastUserQuestion = "";
let messageCounter = 0;
let typingInProgress = false;

const policyDialog = document.getElementById("policyDialog");
const viewPolicyBtn = document.getElementById("viewPolicyBtn");
const closeSurveyBtn = document.getElementById("closeSurveyBtn");
const closePolicyBtn = document.getElementById("closePolicyBtn");
const themeToggleBtn = document.getElementById("themeToggleBtn");
const sidebarToggleBtn = document.getElementById("sidebarToggleBtn");
const sampleQuestionsDiv = document.getElementById("sampleQuestions");

const LOCAL_LINK = "http://localhost:5005/webhooks/rest/webhook";
const PROD_LINK =
  "https://rasa-chatbot-721902099793.us-east1.run.app/webhooks/rest/webhook";
const KEYWORD_LINK = "https://api-721902099793.us-east1.run.app/extract_keyword"
const CENSOR_LINK = "https://api-721902099793.us-east1.run.app/censor_text"

let userScrolling = false;
let scrollTimeout = null;

function scrollChatToBottom() {
  const scrollElement = chatLog.closest('#chatContainer');
  if (userScrolling) {
    return;
  }
  if (scrollElement && typeof scrollElement.scrollHeight !== 'undefined') {
    scrollElement.scrollTo({
      top: scrollElement.scrollHeight,
      behavior: 'smooth'
    });
  } else {
    console.warn("Scrollable element not available or ready for scrolling.");
  }
}

function handleScroll() {
  userScrolling = true;
  if (scrollTimeout) {
    clearTimeout(scrollTimeout);
  }
  scrollTimeout = setTimeout(() => {
    userScrolling = false;
  }, 250);
}

function openModal(modal) {
  if (modal) {
    modal.style.display = "flex";
    const focusable = modal.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
    if (focusable) {
      focusable.focus();
    }
  }
}

function closeModal(modal) {
  if (modal) {
    modal.style.display = "none";
  }
}

function wrapTextNodes(node, charSpans) {
  if (node.nodeType === Node.TEXT_NODE) {
    const text = node.nodeValue;
    const fragment = document.createDocumentFragment();
    for (let i = 0; i < text.length; i++) {
      const span = document.createElement('span');
      span.className = 'typewriter-char';
      span.textContent = text[i];

      if (text[i] === ' ' && (i > 0 && text[i - 1] === ' ' || i < text.length - 1 && text[i + 1] === ' ')) {
        span.innerHTML = ' ';
      } else if (text[i] === '\n') {
        span.textContent = '\n';
      }
      fragment.appendChild(span);
      charSpans.push(span);
    }
    try {
      if (node.parentNode) {
        node.parentNode.replaceChild(fragment, node);
      }
    } catch (e) {
      console.error("Error replacing text node:", e, node);
    }
  } else if (node.nodeType === Node.ELEMENT_NODE && node.childNodes.length > 0) {
    const children = Array.from(node.childNodes);
    children.forEach(child => wrapTextNodes(child, charSpans));
  }
}


function typeWriterEffect(element, finalHtml, speed = 10) {
  if (!element) {
    console.error("Typewriter effect called on a null or undefined element.");
    return;
  }
  if (typingInProgress) {
    console.warn("Typing effect already running. Overlapping effects might occur.");
  }
  typingInProgress = true;
  element.innerHTML = finalHtml;
  element.classList.add("typing");

  const charSpans = [];
  wrapTextNodes(element, charSpans);

  let i = 0;
  let initialScrollHeight = 0;
  try {
    initialScrollHeight = element.scrollHeight;
  } catch (e) {
    console.warn("Could not read scrollHeight during typewriter init", e);
  }


  function type() {
    if (i < charSpans.length) {
      const span = charSpans[i];
      if (span) {
        span.style.opacity = '1';
      } else {
        console.warn("Missing span at index", i, "during typing.");
      }
      i++;

      if (element.closest) {
        const isLastMessageContainer = element.closest('.message-container') === chatLog?.lastElementChild;
        const isNearBottom = chatLog && (chatLog.scrollHeight - chatLog.scrollTop <= chatLog.clientHeight + 100); // Threshold

        if (isLastMessageContainer || isNearBottom) {
          try {
            scrollChatToBottom();
          } catch (e) { }
        }
      }

      setTimeout(type, speed);
    } else {
      if (element) {
        element.classList.remove("typing");
      }
      typingInProgress = false;
      setTimeout(scrollChatToBottom, 50);

      const msgContainer = element.closest(".message-container");
      const controlsDivMsg = msgContainer?.querySelector('.message-controls');
      const feedbackDiv = msgContainer?.querySelector(".feedback-buttons");


      if (feedbackDiv) feedbackDiv.style.display = "block";


      if (element?.closest) {
        const messageDiv = element.closest('.message');
        const showMoreBtnMsg = controlsDivMsg?.querySelector('.show-more-btn');

        if (messageDiv && controlsDivMsg && showMoreBtnMsg && element.classList.contains('message-content-preview')) {
          try {
            if (element.scrollHeight > element.clientHeight + 10) {
              showMoreBtnMsg.style.display = 'inline-block';
              controlsDivMsg.style.display = 'block';
            } else {
              showMoreBtnMsg.style.display = 'none';
              controlsDivMsg.style.display = 'none';
            }
          } catch (e) { }
        }

        const optionCard = element.closest('.option-card');
        const optionControls = optionCard?.querySelector('.option-controls');
        const showMoreBtnOption = optionControls?.querySelector('.show-more-btn');

        if (optionCard && optionControls && showMoreBtnOption && element.classList.contains('option-content-preview')) {
          try {
            if (element.scrollHeight > element.clientHeight + 10) {
              showMoreBtnOption.style.display = 'inline-block';
              optionControls.style.display = 'flex';
            } else {
              showMoreBtnOption.style.display = 'none';
            }
          } catch (e) { }
        }
      }

      setTimeout(scrollChatToBottom, 50);
    }
  }
  type();
}

function getTextContentFromHtml(htmlString) {
  const tempDiv = document.createElement('div');
  tempDiv.innerHTML = htmlString;
  return tempDiv.textContent || tempDiv.innerText || "";
}

// Output message to chat - user or bot side
function addMessageToChat(text, ...classNames) {
  // Create container for the message
  const container = document.createElement("div");
  container.classList.add("message-container");

  // Create the message div
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message", ...classNames);

  // Set the message text
  messageDiv.textContent = text;

  // Append message to container
  container.appendChild(messageDiv);

  // Append container to chat log
  chatLog.appendChild(container);

  // Scroll to bottom
  setTimeout(scrollChatToBottom, 0);
}

viewPolicyBtn.addEventListener('click', () => openModal(policyDialog));
closeSurveyBtn.addEventListener('click', () => closeModal(survey));
closePolicyBtn.addEventListener('click', () => closeModal(policyDialog));

window.addEventListener('click', (event) => {
  if (event.target.classList.contains('modal')) {
    closeModal(event.target);
  }
});

window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    const openModalElement = document.querySelector('.modal[style*="display: flex"]');
    if (openModalElement) {
      closeModal(openModalElement);
    }
  }
});

// Message feedback -- would need reimplemented if this is an aspect we are handling 
//  -- Steven D.
function handleFeedbackClick(messageId, feedbackType) {
  const messageContainer = document.querySelector(`[data-message-id="${messageId}"]`)?.closest('.message-container');
  if (!messageContainer) return;

  const feedbackWrapper = messageContainer.querySelector(".feedback-input-wrapper");
  const textInput = messageContainer.querySelector(".feedback-text-input");
  const targetBtn = messageContainer.querySelector(feedbackType === "positive" ? ".thumbs-up" : ".thumbs-down");
  const thankYouMsg = messageContainer.querySelector(".feedback-thank-you");
  const submitBtn = messageContainer.querySelector(".submit-feedback-btn");

  if (!feedbackWrapper || !textInput || !targetBtn || !thankYouMsg || !submitBtn) {
    console.error("Feedback elements not found for message:", messageId);
    return;
  }


  thankYouMsg.style.display = 'none';

  if (targetBtn.classList.contains("selected")) {
    targetBtn.classList.remove("selected");
    feedbackWrapper.style.opacity = "0";
    feedbackWrapper.style.height = "0";
    feedbackWrapper.style.overflow = "hidden";
    submitBtn.onclick = null;
    return;
  }

  const buttons = messageContainer.querySelectorAll(".feedback-btn");
  buttons.forEach((btn) => btn.classList.remove("selected"));
  targetBtn.classList.add("selected");

  feedbackWrapper.style.opacity = "1";
  feedbackWrapper.style.height = "auto";
  feedbackWrapper.style.overflow = "visible";
  textInput.value = '';

  textInput.className = 'feedback-text-input'; // Reset classes
  textInput.classList.add(feedbackType === "positive" ? "positive" : "negative");
  textInput.placeholder = feedbackType === "positive" ? "What did you like? (Optional)" : "What went wrong? (Optional)";

  textInput.focus();
  // setTimeout(scrollChatToBottom, 50);

  submitBtn.onclick = () => submitMessageFeedback(messageId, feedbackType, textInput.value);
}


function submitMessageFeedback(messageId, feedback, feedbackText) {
  if (!feedback) return;

  const messageElement = document.querySelector(`[data-message-id="${messageId}"]`);
  if (!messageElement) {
    console.error("Could not find message element for feedback:", messageId);
    return;
  }
  const messageContentElement = messageElement.querySelector('.message-content-preview');
  const messageHtml = messageContentElement ? messageContentElement.innerHTML : messageElement.innerHTML;
  const messageText = getTextContentFromHtml(messageHtml);

  const messageContainer = messageElement.closest('.message-container');
  const userQuestion = messageContainer ? messageContainer.getAttribute("data-user-question") || "" : "";

  const feedbackData = {
    feedback: feedback,
    feedbackText: feedbackText.trim() || "",
    question: userQuestion,
    response: messageText.trim(),
    timestamp: new Date().toISOString(),
    sender: senderID
  };

  // removed firebase saving of feedback

}


function submitFeedback() {
  const rating = ratingInput.value;
  const comments = commentsInput.value.trim();

  if (!ratingInput.reportValidity()) {
    return;
  }

  const feedbackData = {
    rating: parseInt(rating, 10),
    feedback: comments,
    sender: senderID,
    timestamp: new Date().toISOString()
  };

  // console.log("Submitting survey feedback:", feedbackData);

  submitFeedbackBtn.disabled = true;
  submitFeedbackBtn.textContent = 'Submitting...';

  // removed firebase submit of feedback

} // End of message feedback


function toggleSidebar() {
  const body = document.body;
  if (!body || !sidebarToggleBtn) return;

  body.classList.toggle('sidebar-collapsed');
  const isCollapsed = body.classList.contains('sidebar-collapsed');
  localStorage.setItem('sidebarCollapsed', isCollapsed);

  if (window.innerWidth > 768) {
    sidebarToggleBtn.textContent = isCollapsed ? '>' : '<';
    sidebarToggleBtn.setAttribute('aria-label', isCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar');
  } else {
    sidebarToggleBtn.textContent = '';
    sidebarToggleBtn.setAttribute('aria-label', isCollapsed ? 'Show Info' : 'Hide Info');
  }
  sidebarToggleBtn.setAttribute('aria-expanded', !isCollapsed);

}


document.addEventListener("DOMContentLoaded", () => {
  if (!document.body || !localStorage || !themeToggleBtn || !sidebarToggleBtn || !chatLog || !userInput || !sendBtn || !policyDialog || !viewPolicyBtn || !survey) {
    console.error("Initialization failed: One or more critical DOM elements are missing.");
    return;
  }

  document.body.classList.remove('preload');

  // sendMessageToBot("/greet");

  const currentTheme = localStorage.getItem('theme') || 'light-mode'; // Default to light
  // document.body.classList.add(currentTheme);
  themeToggleBtn.textContent = currentTheme === 'dark-mode' ? '🌙' : '☀️';

  // Add event listener for theme toggle button
  // themeToggleBtn.addEventListener('click', toggleTheme);

  // Ensure sample questions are visible on load if chat is empty (assuming /greet is the only initial message)
  if (chatLog.children.length <= 1 && sampleQuestionsDiv) { // Check if chatLog has only the greet message or is empty
    sampleQuestionsDiv.style.display = 'block';
  }

  sidebarToggleBtn.addEventListener('click', toggleSidebar);

  chatLog.addEventListener('click', (event) => {
    const link = event.target.closest('a');
    if (link && link.href) {
      if (link.hostname !== window.location.hostname || link.pathname !== window.location.pathname || !link.target) {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
      }
    }
  });

  chatLog.addEventListener('scroll', handleScroll);


  window.addEventListener('resize', () => {
    // Only update button appearance on resize, not layout
    const isCollapsed = document.body.classList.contains('sidebar-collapsed');
    if (window.innerWidth <= 768) {
      sidebarToggleBtn.textContent = '';
      sidebarToggleBtn.setAttribute('aria-label', isCollapsed ? 'Show Info' : 'Hide Info');
    } else {
      sidebarToggleBtn.textContent = isCollapsed ? '>' : '<';
      sidebarToggleBtn.setAttribute('aria-label', isCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar');
    }
  });
});

// Quick action buttons
document.querySelectorAll(".actionBtn").forEach((button) => {
  button.addEventListener("click", () => {
    const query = button.dataset.query;
    if (userInput) { // Check if userInput exists
      userInput.value = query;
      sendMessageToBot(query);
      userInput.value = ''; // Clear after sending
    }
  });
});

// --- sendMessageToBot (previously sendMessageToRasa) ---
// ---- figure out hosting
async function sendMessageToBot(message) {
  message = message?.trim();
  if (!message) return;

  addMessageToChat(message, "userMsg");

  if (userInput) userInput.value = "";

  // Hide sample questions after the first user message
  if (sampleQuestionsDiv) {
    sampleQuestionsDiv.style.display = 'none';
  }

  // "Bot" reply
  addMessageToChat("Forced test response", "botMsg");

  return;
}

function showSurvey() {
  openModal(survey);
}

sendBtn.addEventListener("click", () => {
  sendMessageToBot(userInput.value);
});

userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { // Send on Enter only
    e.preventDefault(); // Prevent default newline on Enter
    sendMessageToBot(userInput.value);
  }
});

submitFeedbackBtn.addEventListener("click", submitFeedback);

document.getElementById("endChatBtn").addEventListener("click", showSurvey);

window.handleFeedbackClick = handleFeedbackClick;
window.toggleTheme = toggleTheme;

function toggleTheme() {
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const body = document.body;

  // Toggle dark mode class
  body.classList.toggle('dark-mode');

  // Update theme toggle button content based on current state
  const isDarkMode = body.classList.contains('dark-mode');
  themeToggleBtn.textContent = isDarkMode ? '🌙' : '☀️';

  // Save theme preference
  localStorage.setItem('theme', isDarkMode ? 'dark-mode' : 'light-mode');
}
import React, { useState } from 'react';
import axios from 'axios';

import './App.css';

function App() {
  const API = 'http://localhost:5000/predict';

  const [message, setMessage] = useState('');
  const [conversation, setConversation] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!message.trim()) return;

    // Add user message to conversation
    setConversation(prev => [...prev, { type: 'user', text: message }]);

    try {
      const result = await axios.post(API, { message });
      // Add API response to conversation
      console.log(result.data)
      setConversation(prev => [...prev, { type: 'bot', text: result.data.response }]);
      setMessage('');
    } catch (error) {
      console.error('Error calling API:', error);
      setConversation(prev => [...prev, { type: 'error', text: 'Error: Could not get response from server' }]);
    }
  };

  return (
    <div className="main">
      <h1>AAI520 - Natural Language Processing</h1>
      <div className="chat">
        {conversation.map((entry, index) => (
          <div className={"message " + entry.type} key={index}>
            <strong>{entry.type === 'user' ? 'You:' : 'Bot:'}</strong> {entry.text}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit}>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your message here"
          rows={4}
          cols={50}
        />
        <br />
        <button type="submit">Send</button>
      </form>
    </div>
  );
}

export default App;

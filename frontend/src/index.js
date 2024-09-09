// src/index.js
import React from 'react';
import ReactDOM from 'react-dom';
import './index.css'; // Importing CSS
import App from './App'; // Importing the App component

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')  // This mounts the app into the root div in index.html
);

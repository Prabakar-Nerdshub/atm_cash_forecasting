import React from 'react';
import './App.css'; // Import the CSS file

const App = () => {
  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <h2>Dashboard</h2>
        <ul>
          <li><a href="#overview">Overview</a></li>
          <li><a href="#analytics">Analytics</a></li>
          <li><a href="#settings">Settings</a></li>
        </ul>
      </aside>
      <div className="main-content">
        <header className="topbar">
          <h1>Dashboard</h1>
        </header>
        <main className="content">
          <section id="overview">
            <h2>Overview</h2>
            <p>This is the overview section.</p>
          </section>
          <section id="analytics">
            <h2>Analytics</h2>
            <p>This is the analytics section.</p>
          </section>
          <section id="settings">
            <h2>Settings</h2>
            <p

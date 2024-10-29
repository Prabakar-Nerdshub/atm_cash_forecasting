import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';

const App = () => {
  const [selectedDataset, setSelectedDataset] = useState('atm_cash_demand_single_atm'); // Set a default dataset
  const [rmse, setRmse] = useState(null); // Store RMSE value
  const [forecastAmount, setForecastAmount] = useState(null); // Store forecast amount
  const [isLoading, setIsLoading] = useState(false); // Loading state
  const [error, setError] = useState(null); // Error state

  // Function to fetch data based on selected dataset
  const fetchData = async () => {
    if (!selectedDataset) return; // Prevent fetch if no dataset is selected

    setIsLoading(true);
    setError(null); // Reset error state before fetch

    try {
      const response = await axios.get(`/api/data/?dataset=${selectedDataset}`);
      const data = response.data;

      setRmse(data.rmse);
      setForecastAmount(data.forecast_amount);
    } catch (error) {
      console.error('Error fetching data:', error);
      setError('Failed to fetch data. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Effect to fetch data when the selected dataset changes
  useEffect(() => {
    fetchData(); // Call fetchData immediately on mount
  }, []); // Empty dependency array to run only on mount

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
          <h1>Forecasting Dashboard</h1>
        </header>

        <main className="content">
          <section id="dataset-selection" className="widget">
            <h2>Data Overview</h2>
            <p>Selected Dataset: {selectedDataset}</p>
            <button onClick={fetchData} disabled={isLoading}>Fetch Forecast Data</button>
          </section>

          {/* Show loading state */}
          {isLoading && <p>Loading...</p>}
          {error && <p className="error">{error}</p>} {/* Display error message */}

          {/* Render the RMSE and Forecast Amount */}
          {rmse !== null && forecastAmount !== null && (
            <div className="widget">
              <h2>Model RMSE Value: {rmse}</h2>
              <h2>Forecasted Amount: {forecastAmount}</h2>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;

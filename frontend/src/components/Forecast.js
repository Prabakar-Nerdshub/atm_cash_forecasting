// src/components/Forecast.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2'; // Assuming you're using Chart.js for rendering the graph

const Forecast = ({ selectedModel }) => {
  const [chartData, setChartData] = useState({ labels: [], values: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchGraphData = async () => {
      try {
        setLoading(true);
        let response;
        switch (selectedModel) {
          case 'xgboost':
            response = await axios.get('/api/xgboost/data/');
            break;
          case 'prophet':
            response = await axios.get('/api/prophet/data/');
            break;
          case 'lstm':
            response = await axios.get('/api/lstm/data/');
            break;
          case 'sarimax':
            response = await axios.get('/api/sarimax/data/');
            break;
          default:
            throw new Error('Invalid model selected');
        }
        setChartData(response.data);
      } catch (err) {
        setError('Error fetching graph data');
      } finally {
        setLoading(false);
      }
    };

    if (selectedModel) {
      fetchGraphData();
    }
  }, [selectedModel]);

  if (loading) {
    return <div>Loading graph data...</div>;
  }

  if (error) {
    return <div>{error}</div>;
  }

  return (
    <div>
      <h2>{selectedModel.toUpperCase()} Forecast Data</h2>
      <Line
        data={{
          labels: chartData.labels,
          datasets: [
            {
              label: 'Forecast Values',
              data: chartData.values,
              fill: false,
              backgroundColor: 'rgba(75,192,192,0.4)',
              borderColor: 'rgba(75,192,192,1)',
            },
          ],
        }}
      />
    </div>
  );
};

export default Forecast;

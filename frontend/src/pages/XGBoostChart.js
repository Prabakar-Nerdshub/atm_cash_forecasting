import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS } from 'chart.js/auto';

const XGBoostChart = () => {
    const [loading, setLoading] = useState(true);
    const [chartData, setChartData] = useState({
        labels: [], // Initialize labels as empty
        datasets: [{
            label: 'Loading data...', // Placeholder while loading
            data: [],
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
        }],
    });

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            try {
                const response = await fetch('your-api-url'); // Replace with actual API URL
                const result = await response.json();

                // Assuming your result object has 'labels' and 'data' properties
                setChartData({
                    labels: result.labels, // Use fetched data for labels
                    datasets: [{
                        label: 'XGBoost Predictions',
                        data: result.data, // Use fetched data for chart
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1,
                    }]
                });
            } catch (error) {
                console.error('Error fetching chart data:', error);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    return (
        <div>
            {loading ? (
                <p>Loading...</p>
            ) : (
                <Line data={chartData} />
            )}
        </div>
    );
};

export default XGBoostChart;

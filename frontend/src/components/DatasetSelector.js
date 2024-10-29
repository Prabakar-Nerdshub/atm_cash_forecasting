import React, { useState } from 'react';

const DatasetSelector = ({ onModelChange }) => {
  const [selectedDataset, setSelectedDataset] = useState('');

  const handleDatasetChange = (event) => {
    const datasetName = event.target.value;
    setSelectedDataset(datasetName);

    // Notify parent about model selection based on dataset (adjust as necessary)
    if (datasetName === 'atm_cash_demand_single_atm') {
      onModelChange('xgboost'); // Set the default model based on dataset
    } else {
      onModelChange(''); // Reset if a different dataset is selected
    }
  };

  return (
    <div>
      <select onChange={handleDatasetChange} value={selectedDataset}>
        <option value="">Select a dataset</option>
        <option value="atm_cash_demand_single_atm">ATM Cash Demand Single ATM</option>
        {/* Add other datasets as needed */}
      </select>
    </div>
  );
};

export default DatasetSelector;

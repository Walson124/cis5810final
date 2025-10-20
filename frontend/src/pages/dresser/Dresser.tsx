import React, { useState } from 'react';
import { Box, Tabs, Tab } from '@mui/material';

// Import your ClothesGallery component (make sure to adjust the path)
import ClothesGallery from './ClothesGallery';
import TryOn from './TryOn';

const Dresser: React.FC = () => {
  const [tabIndex, setTabIndex] = useState(0);

  const handleChange = (_event: React.SyntheticEvent, newIndex: number) => {
    setTabIndex(newIndex);
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Tabs value={tabIndex} onChange={handleChange} aria-label="Dresser Tabs">
        <Tab label="Clothes Gallery" />
        <Tab label="Try On" />
      </Tabs>

      <Box sx={{ mt: 2 }}>
        {tabIndex === 0 && <ClothesGallery />}
        {tabIndex === 1 && <TryOn />}
      </Box>
    </Box>
  );
};

export default Dresser;

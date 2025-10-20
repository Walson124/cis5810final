import React, { useState } from 'react';
import { Box, Button, ButtonGroup, ImageList, ImageListItem } from '@mui/material';

const dummyClothes = [
  { id: 1, type: 'shirt', img: 'src/assets/img/dummy/9761.jpg' },
  { id: 2, type: 'pants', img: 'src/assets/img/dummy/9776.jpg' },
  { id: 3, type: 'jacket', img: 'src/assets/img/dummy/9874.jpg' },
  { id: 4, type: 'shirt', img: 'src/assets/img/dummy/9842.jpg' },
  { id: 5, type: 'pants', img: 'src/assets/img/dummy/9852.jpg' },
];

const allTypes = ['all', 'shirt', 'pants', 'jacket'];

export default function ClothesGallery() {
  const [filter, setFilter] = useState('all');

  const filteredClothes = filter === 'all'
    ? dummyClothes
    : dummyClothes.filter(item => item.type === filter);

  return (
    <Box>
      <ButtonGroup sx={{ mb: 2 }}>
        {allTypes.map((type) => (
          <Button
            key={type}
            variant={filter === type ? 'contained' : 'outlined'}
            onClick={() => setFilter(type)}
          >
            {type.charAt(0).toUpperCase() + type.slice(1)}
          </Button>
        ))}
      </ButtonGroup>

      <ImageList cols={3} gap={12}>
        {filteredClothes.map(({ id, img, type }) => (
          <ImageListItem key={id}>
            <img
              src={img}
              alt={`${type} ${id}`}
              loading="lazy"
              style={{ borderRadius: 8, width: '100%', height: 'auto' }}
            />
          </ImageListItem>
        ))}
      </ImageList>
    </Box>
  );
}

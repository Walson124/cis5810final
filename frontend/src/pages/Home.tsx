import React, { useEffect, useState } from 'react';
import Header from '../components/Header';
import Dresser from './dresser/Dresser';
import About from './about/About';
import { Box } from '@mui/material';
import ImageGallery from '../components/ImageGallery/ImageGallery';

const Home: React.FC = () => {
	const [currentPage, setCurrentPage] = useState("Home");
	const [pageOptions] = useState([
		"Home",
		"My Dresser",
		"About Us",
	]);

	const [query, setQuery] = useState("modern");

	return (
		<div>
			<Header
				pageName={currentPage}
				setCurrentPage={setCurrentPage}
				pageOptions={pageOptions}
				setQuery={setQuery}
			/>
			<Box
				sx={{
					padding: '0 3px 3px 3px'
				}}
			>
				{currentPage == "Home" && <ImageGallery query={query}/> }
				{currentPage == "My Dresser" && <Dresser />}
				{currentPage == "About Us" && <About />}
			</Box>
		</div>
	);
};

export default Home;

// import React, { useEffect, useState, useRef } from 'react';
// import { CircularProgress } from '@mui/material';
// import './ImageGallery.css';

// // Assuming this is correctly set up in your environment
// const PIXABAY_API_KEY = import.meta.env.VITE_PIXABAY_API_KEY;

// const ImageGallery: React.FC<{ query: string }> = ({ query }) => {
//   const [images, setImages] = useState<string[]>([]);
//   const [page, setPage] = useState(1);
//   const [loading, setLoading] = useState(false);
//   const [hasMore, setHasMore] = useState(true); // Track if there are more results
//   const observer = useRef<IntersectionObserver | null>(null);
//   const lastImageRef = useRef<HTMLDivElement | null>(null);

//   // ------------------------------------------------------------------
//   // 1. EFFECT TO HANDLE NEW SEARCH QUERY: Reset images and page
//   // ------------------------------------------------------------------
//   useEffect(() => {
//     // When the query changes, reset the state to start a new search
//     setImages([]);
//     setPage(1);
//     setHasMore(true);
//     // Note: The page=1 change will trigger the fetch effect below
//   }, [query]);

//   const lastFetchedPage = useRef<number>(0);

//   useEffect(() => {
//     if (!query || !hasMore) return;

//     // Avoid fetching if we already fetched this page for this query
//     if (lastFetchedPage.current === page) return;

//     lastFetchedPage.current = page;

//     const fetchImages = async () => {
//       setLoading(true);
//       let url = `https://pixabay.com/api/?key=${PIXABAY_API_KEY}&q=${encodeURIComponent(query + ' outfits').replace(/%20/g, '+')}&category=fashion&image_type=photo&page=${page}&per_page=20`
//       console.log(url);
//       const res = await fetch(
//         url
//       );
//       const data = await res.json();

//       if (data.hits.length === 0) setHasMore(false);

//       setImages(prev => [...prev, ...data.hits.map((hit: any) => hit.webformatURL)]);
//       setLoading(false);
//     };

//     fetchImages();

//   }, [page, query, hasMore]);

//   // ------------------------------------------------------------------
//   // 3. EFFECT FOR INTERSECTION OBSERVER: Handles infinite scroll
//   // ------------------------------------------------------------------
//   useEffect(() => {
//     if (loading || !hasMore) return; // Don't attach observer if loading or no more images

//     // Cleanup previous observer
//     if (observer.current) observer.current.disconnect();

//     observer.current = new IntersectionObserver(
//       (entries) => {
//         // If the sentinel div is intersecting, increment the page number
//         if (entries[0].isIntersecting) {
//           setPage((prev) => prev + 1);
//         }
//       },
//       { threshold: 1 }
//     );

//     // Start observing the sentinel div if it exists
//     if (lastImageRef.current) {
//       observer.current.observe(lastImageRef.current);
//     }

//     // Cleanup on unmount
//     return () => {
//       if (observer.current) {
//         observer.current.disconnect();
//       }
//     };

//   }, [loading, hasMore]); // Re-run when loading state changes or hasMore changes

//   return (
//     <>
//       <div className="image-gallery">
//         {images.map((url, index) => (
//           <img
//             key={url + index}
//             src={url}
//             alt={`Image ${index}`}
//             loading="lazy"
//           />
//         ))}
//       </div>

//       {/* Sentinel div for infinite scroll: placed after the last image */}
//       {hasMore && !loading && (
//         <div ref={lastImageRef} style={{ height: '10px' }} />
//       )}

//       {/* Loading spinner */}
//       {loading && (
//         <div style={{ textAlign: 'center', padding: 16 }}>
//           <CircularProgress />
//         </div>
//       )}

//       {/* Message if no more results */}
//       {!hasMore && images.length > 0 && (
//         <div style={{ textAlign: 'center', padding: 16, color: 'gray' }}>
//           End of results for "{query}"
//         </div>
//       )}

//       {/* Message if initial search returned no results */}
//       {!loading && images.length === 0 && query && (
//         <div style={{ textAlign: 'center', padding: 16, color: 'gray' }}>
//           No images found for "{query}"
//         </div>
//       )}
//     </>
//   );
// };

// export default ImageGallery;


import React, { useEffect, useState, useRef } from 'react';
import { CircularProgress } from '@mui/material';
import './ImageGallery.css';

// --- Interface for richer data structure ---
interface PixabayHit {
  id: number;
  webformatURL: string;
  largeImageURL: string;
  tags: string;
  user: string;
  downloads: number;
  likes: number;
  views: number;
}

// --- API Key (Assumed to be correctly set up) ---
const PIXABAY_API_KEY = import.meta.env.VITE_PIXABAY_API_KEY;

// --- Styles object for the modal (reusing the clean, structured styles) ---
const modalStyles = {
  // Modal Backdrop
  modalBackdrop: {
    position: 'fixed' as 'fixed',
    inset: 0,
    background: 'rgba(0, 0, 0, 0.85)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 9999,
    animation: 'fadeIn 0.3s ease-out',
    padding: '1.5rem',
  },
  
  // Modal Content
  modalContent: {
    background: '#fff',
    borderRadius: 12,
    padding: '2rem',
    maxWidth: '90vw',
    maxHeight: '90vh',
    overflowY: 'auto' as 'auto',
    boxShadow: '0 10px 30px rgba(0, 0, 0, 0.4)',
    position: 'relative' as 'relative',
    width: '100%',
  },
  
  // Close Button
  closeButton: {
    position: 'absolute' as 'absolute',
    top: 10,
    right: 10,
    background: 'transparent',
    border: 'none',
    fontSize: '2rem',
    cursor: 'pointer',
    color: '#999',
    lineHeight: 1,
    padding: 0,
    fontWeight: '300' as '300',
    transition: 'color 0.2s ease',
  },

  // Image
  image: {
    maxWidth: '100%',
    maxHeight: '60vh', 
    width: 'auto',
    objectFit: 'contain' as 'contain',
    borderRadius: 8,
    marginBottom: '1.5rem',
    display: 'block',
    marginLeft: 'auto',
    marginRight: 'auto',
    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
  },

  // Details Container
  detailsContainer: {
    lineHeight: 1.6,
    color: '#444',
    paddingTop: '0.5rem',
  },

  // Tags Heading
  tagsHeading: {
    marginTop: 0,
    marginBottom: 16,
    fontWeight: '600' as '600',
    fontSize: '1.1rem',
    color: '#333',
    borderBottom: '1px solid #eee',
    paddingBottom: '8px',
  },
  
  // Tags Value
  tagsValue: {
    fontWeight: '400' as '400', 
    marginLeft: 10,
    color: '#666',
    fontSize: '1rem',
  },

  // Stats List
  statsList: {
      listStyle: 'none', 
      padding: 0,
      margin: 0,
      display: 'grid' as 'grid', 
      gridTemplateColumns: '1fr 1fr',
      gap: '10px 20px', 
  },

  // Stat Item
  statItem: {
      margin: '0',
      fontSize: '0.95rem',
      padding: '4px 0',
      color: '#333',
  }
};


const ImageGallery: React.FC<{ query: string }> = ({ query }) => {
  // --- STATE ---
  const [images, setImages] = useState<PixabayHit[]>([]);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [selectedImage, setSelectedImage] = useState<PixabayHit | null>(null);

  // --- REFS ---
  const observer = useRef<IntersectionObserver | null>(null);
  const lastImageRef = useRef<HTMLDivElement | null>(null);
  const lastFetchedPage = useRef<number>(0);

  // --- MODAL HANDLERS ---
  const openModal = (image: PixabayHit) => setSelectedImage(image);
  const closeModal = () => setSelectedImage(null);

  // ------------------------------------------------------------------
  // 1. EFFECT TO HANDLE NEW SEARCH QUERY: Reset images and page
  // ------------------------------------------------------------------
  useEffect(() => {
    // When the query changes, reset the state to start a new search
    setImages([]);
    setPage(1);
    setHasMore(true);
    lastFetchedPage.current = 0; // Crucial: Reset the page tracker
    closeModal(); // Close modal if open on new search
  }, [query]);

  // ------------------------------------------------------------------
  // 2. EFFECT FOR FETCHING IMAGES
  // ------------------------------------------------------------------
  useEffect(() => {
    // ⚠️ Prevent fetching if no query, no more results, or on first render (page is 1, but images might not be empty yet)
    if (!query || !hasMore || (page === 1 && images.length > 0)) {
        if (!query) setImages([]);
        return;
    }

    // Debounce/guard against double-fetches
    if (lastFetchedPage.current === page) return;

    const fetchImages = async () => {
      // Set loading state and update the tracker immediately
      setLoading(true);
      lastFetchedPage.current = page;

      try {
        const encodedQuery = encodeURIComponent(query + ' outfits').replace(/%20/g, '+');
        const url = `https://pixabay.com/api/?key=${PIXABAY_API_KEY}&q=${encodedQuery}&category=fashion&image_type=photo&page=${page}&per_page=20`;
        
        const res = await fetch(url);
        if (!res.ok) throw new Error('Network response was not ok');
        
        const data = await res.json();

        // Pixabay's total hits can be used, but checking the hits array is simpler
        if (data.hits.length === 0) {
          setHasMore(false);
        }

        setImages(prev => {
          // Filter out duplicates based on ID (optional, but good practice)
          const newHits = data.hits.filter((hit: PixabayHit) => !prev.some(p => p.id === hit.id));
          return [...prev, ...newHits];
        });

      } catch (error) {
        console.error("Failed to fetch images:", error);
        // Optionally show an error message to the user
      } finally {
        setLoading(false);
      }
    };

    fetchImages();

  }, [page, query, hasMore, images.length]); // Added images.length to dependency array to check against page=1 condition

  // ------------------------------------------------------------------
  // 3. EFFECT FOR INTERSECTION OBSERVER: Handles infinite scroll
  // ------------------------------------------------------------------
  useEffect(() => {
    // Don't attach observer if loading, no more images, or if no images are currently displayed
    if (loading || !hasMore || images.length === 0) return;

    // Cleanup previous observer
    if (observer.current) observer.current.disconnect();

    observer.current = new IntersectionObserver(
      ([entry]) => {
        // If the sentinel div is intersecting, and we're not already loading
        if (entry.isIntersecting && !loading) {
          setPage((prev) => prev + 1);
        }
      },
      { threshold: 1 }
    );

    // Start observing the sentinel div if it exists
    if (lastImageRef.current) {
      observer.current.observe(lastImageRef.current);
    }

    // Cleanup on unmount
    return () => {
      if (observer.current) {
        observer.current.disconnect();
      }
    };

  }, [loading, hasMore, images.length]);


  return (
    <>
      <div className="image-gallery">
        {images.map((image, index) => (
          // Use image.id for key for better stability
          <img
            key={image.id} 
            src={image.webformatURL}
            alt={image.tags || `Image ${index}`}
            loading="lazy"
            onClick={() => openModal(image)} // 👈 OPEN MODAL ON CLICK
          />
        ))}

        {/* --- Infinite Scroll Sentinel --- */}
        {hasMore && images.length > 0 && (
          <div ref={lastImageRef} style={{ height: '10px' }} />
        )}
      </div>

      {/* --- Loading and End-of-Results Messages --- */}
      {loading && (
        <div style={{ textAlign: 'center', padding: 16 }}>
          <CircularProgress />
        </div>
      )}

      {!hasMore && images.length > 0 && (
        <div style={{ textAlign: 'center', padding: 16, color: 'gray' }}>
          End of results for "{query}" 🔚
        </div>
      )}

      {/* Message if initial search returned no results */}
      {!loading && images.length === 0 && query && (
        <div style={{ textAlign: 'center', padding: 16, color: 'gray' }}>
          No images found for "{query}" 😔. Try a different query!
        </div>
      )}

      {/* ------------------------------------------------------------------
        // 4. MODAL COMPONENT (Using selectedImage state)
        // ------------------------------------------------------------------ */}
      {selectedImage && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Image details"
          style={modalStyles.modalBackdrop}
          onClick={closeModal}
        >
          <div
            style={modalStyles.modalContent}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close Button */}
            <button
              onClick={closeModal}
              aria-label="Close image details modal"
              style={modalStyles.closeButton}
            >
              &times;
            </button>

            {/* Image */}
            <img
              src={selectedImage.largeImageURL || selectedImage.webformatURL}
              alt={selectedImage.tags}
              style={modalStyles.image}
            />

            {/* Details Section */}
            <div style={modalStyles.detailsContainer}>
              <h3 style={modalStyles.tagsHeading}>
                Tags:
                <span style={modalStyles.tagsValue}>{selectedImage.tags}</span>
              </h3>
              
              <ul style={modalStyles.statsList}>
                <li style={modalStyles.statItem}>
                  <strong>By:</strong> {selectedImage.user}
                </li>
                <li style={modalStyles.statItem}>
                  <strong>Downloads:</strong> {selectedImage.downloads.toLocaleString()}
                </li>
                <li style={modalStyles.statItem}>
                  <strong>Likes:</strong> {selectedImage.likes.toLocaleString()}
                </li>
                <li style={modalStyles.statItem}>
                  <strong>Views:</strong> {selectedImage.views.toLocaleString()}
                </li>
              </ul>
            </div>
          </div>
          
          {/* CSS for the fade-in animation */}
          <style>
            {`
              @keyframes fadeIn {
                from {opacity: 0;}
                to {opacity: 1;}
              }
            `}
          </style>
        </div>
      )}
    </>
  );
};

export default ImageGallery;
import React from 'react';
import headerBg from '../../assets/img/dummy/gpt_img.png';

const TryOn: React.FC = () => {
    return (
        <div>
            <img src={headerBg} alt="Header Background" style={{ maxWidth: '100%', height: 'auto' }} />
        </div>
    );
};

export default TryOn;

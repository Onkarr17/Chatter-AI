import React from 'react';

const Loader: React.FC<{ size?: number; color?: string }> = ({ size = 24, color = 'var(--accent-cyan)' }) => {
    return (
        <div className="loader" style={{
            display: 'inline-block',
            width: `${size}px`,
            height: `${size}px`,
            border: `3px solid rgba(255, 255, 255, 0.1)`,
            borderTop: `3px solid ${color}`,
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
        }} />
    );
};

export default Loader;

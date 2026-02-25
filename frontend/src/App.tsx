import React, { useState } from 'react';
import Layout from './components/Layout';
import { ThemeProvider } from './context/ThemeContext';
import { ChatProvider } from './context/ChatContext';
import ChatPage from './pages/ChatPage';
import DocumentsPage from './pages/DocumentsPage';
import AnalyticsPage from './pages/AnalyticsPage';

const App: React.FC = () => {
    const [activeTab, setActiveTab] = useState<'chat' | 'docs' | 'analytics'>('chat');

    const renderContent = () => {
        switch (activeTab) {
            case 'chat':
                return <ChatPage />;
            case 'docs':
                return <DocumentsPage />;
            case 'analytics':
                return <AnalyticsPage />;
            default:
                return <ChatPage />;
        }
    };

    return (
        <ThemeProvider>
            <ChatProvider>
                <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
                    {renderContent()}
                </Layout>
            </ChatProvider>
        </ThemeProvider>
    );
};

export default App;

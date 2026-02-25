import React, { createContext, useContext, useState, useEffect } from 'react';
import { chatApi } from '../api/chat.api';
import { documentsApi } from '../api/documents.api';
import { QueryResponse } from '../types';

interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    similarity?: number;
}

interface ChatContextType {
    messages: ChatMessage[];
    selectedFile: File | null;
    setSelectedFile: (file: File | null) => void;
    question: string;
    setQuestion: (q: string) => void;
    loading: boolean;
    error: string | null;
    handleAsk: () => Promise<void>;
    clearChat: () => void;
    hasActiveDocs: boolean;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const ChatProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [question, setQuestion] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [hasActiveDocs, setHasActiveDocs] = useState(false);

    // Check for active documents to enable chat
    const checkDocs = async () => {
        try {
            const docs = await documentsApi.getDocuments();
            setHasActiveDocs(docs.length > 0);
        } catch (err) {
            console.error('Failed to check documents', err);
        }
    };

    useEffect(() => {
        checkDocs();
        // Also re-check periodically or could be triggered on refocus/tab change
        const interval = setInterval(checkDocs, 5000);
        return () => clearInterval(interval);
    }, []);

    const handleAsk = async () => {
        // Enable if either a file is selected OR the system has active docs
        if (!selectedFile && !hasActiveDocs) {
            setError('Please upload a PDF in the Documents tab first.');
            return;
        }

        if (!question.trim()) return;

        const userMsg: ChatMessage = { role: 'user', content: question };
        setMessages(prev => [...prev, userMsg]);
        setLoading(true);
        setError(null);
        const currentQuestion = question;
        setQuestion('');

        try {
            let response: QueryResponse;

            if (selectedFile) {
                // RULE 1: If user uploads/selects a PDF -> Call /query
                response = await chatApi.submitQuery(selectedFile, currentQuestion);
            } else {
                // RULE 2: Else (Chat with Library/Auto-indexed docs) -> Call /chat/ask
                response = await chatApi.askQuestion(currentQuestion);
            }

            const assistantMsg: ChatMessage = {
                role: 'assistant',
                content: response.answer,
                similarity: response.best_similarity
            };
            setMessages(prev => [...prev, assistantMsg]);
        } catch (err: any) {
            console.error('Chat Error:', err);
            const errorMsg = err.response?.data?.detail;

            if (typeof errorMsg === 'string') {
                setError(errorMsg);
            } else if (typeof errorMsg === 'object' && errorMsg !== null) {
                // Handle FastAPI validation error objects or other structured errors
                setError(JSON.stringify(errorMsg));
            } else {
                setError('Failed to get an answer. Please check your connection.');
            }
        } finally {
            setLoading(false);
        }
    };

    const clearChat = () => setMessages([]);

    return (
        <ChatContext.Provider value={{
            messages,
            selectedFile,
            setSelectedFile,
            question,
            setQuestion,
            loading,
            error,
            handleAsk,
            clearChat,
            hasActiveDocs
        }}>
            {children}
        </ChatContext.Provider>
    );
};

export const useChatContext = () => {
    const context = useContext(ChatContext);
    if (!context) throw new Error('useChatContext must be used within a ChatProvider');
    return context;
};

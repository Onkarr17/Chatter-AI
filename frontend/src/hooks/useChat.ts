import { useState } from 'react';
import { chatApi } from '../api/chat.api';
import { QueryResponse } from '../types';

export const useChat = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState<string | null>(null);
    const [similarity, setSimilarity] = useState<number | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleAsk = async () => {
        if (!selectedFile || !question.trim()) {
            setError('Please upload a PDF and enter a question.');
            return;
        }

        setLoading(true);
        setError(null);
        setAnswer(null);
        setSimilarity(null);

        try {
            const response: QueryResponse = await chatApi.submitQuery(selectedFile, question);
            setAnswer(response.answer);
            setSimilarity(response.best_similarity);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to get an answer. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return {
        selectedFile,
        setSelectedFile,
        question,
        setQuestion,
        answer,
        similarity,
        loading,
        error,
        handleAsk,
    };
};

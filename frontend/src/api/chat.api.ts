import client from './client';
import { QueryResponse } from '../types';

export const chatApi = {
    submitQuery: async (file: File | null, question: string): Promise<QueryResponse> => {
        const formData = new FormData();
        if (file) {
            formData.append('file', file);
        }
        formData.append('question', question);

        const { data } = await client.post<QueryResponse>('/query', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return data;
    },
    askQuestion: async (question: string): Promise<QueryResponse> => {
        const formData = new FormData();
        formData.append('question', question);

        const { data } = await client.post<QueryResponse>('/chat/ask', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return data;
    },
};

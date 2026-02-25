import client from './client';
import { Document } from '../types';

export const documentsApi = {
    getDocuments: async (): Promise<Document[]> => {
        // Ensure trailing slash for consistent backend routing
        const { data } = await client.get<Document[]>('/documents/');
        return data;
    },
    deleteDocument: async (documentId: string): Promise<void> => {
        await client.delete(`/documents/${documentId}`);
    },
};

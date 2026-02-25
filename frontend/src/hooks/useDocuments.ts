import { useState, useEffect, useCallback } from 'react';
import { documentsApi } from '../api/documents.api';
import { Document } from '../types';

export const useDocuments = () => {
    const [documents, setDocuments] = useState<Document[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchDocuments = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await documentsApi.getDocuments();
            setDocuments(data);
        } catch (err: any) {
            setError('Failed to load documents.');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchDocuments();
    }, [fetchDocuments]);

    const removeDocument = async (id: string) => {
        // Optimistic Update: Remove from UI immediately
        const previousDocs = documents;
        setDocuments(prev => prev.filter(d => d.document_id !== id));

        try {
            await documentsApi.deleteDocument(id);
        } catch (err: any) {
            // Rollback on error
            setDocuments(previousDocs);
            setError('Failed to delete document from the server.');
        }
    };

    return {
        documents,
        loading,
        error,
        refresh: fetchDocuments,
        removeDocument,
    };
};

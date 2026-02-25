export interface Document {
    document_id: string;
    filename: string;
    total_pages: number;
    created_at: string;
}

export interface QueryResponse {
    answer: string;
    best_similarity: number;
}

export interface AnalyticsSummary {
    total_documents: number;
    total_queries: number;
    average_similarity: number;
    most_queried_document: {
        filename: string;
        query_count: number;
    } | null;
}

export type Theme = 'dark' | 'light';

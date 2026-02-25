import client from './client';
import { AnalyticsSummary } from '../types';

export const analyticsApi = {
    getSummary: async (): Promise<AnalyticsSummary> => {
        // Fixed path from /analytics/summary to /analytics/overview
        const { data } = await client.get<AnalyticsSummary>('/analytics/overview');
        return data;
    },
};

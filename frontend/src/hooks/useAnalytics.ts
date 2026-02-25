import { useState, useEffect } from 'react';
import { analyticsApi } from '../api/analytics.api';
import { AnalyticsSummary } from '../types';

export const useAnalytics = () => {
    const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchAnalytics = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await analyticsApi.getSummary();
            setSummary(data);
        } catch (err: any) {
            setError('Failed to load analytics.');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchAnalytics();
    }, []);

    return {
        summary,
        loading,
        error,
        refresh: fetchAnalytics,
    };
};

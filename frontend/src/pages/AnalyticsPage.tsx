import React from 'react';
import { useAnalytics } from '../hooks/useAnalytics';
import { Files, MessageSquare, Target, Trophy, TrendingUp, BarChart } from 'lucide-react';
import Loader from '../components/common/Loader';

const AnalyticsPage: React.FC = () => {
    const { summary, loading, error } = useAnalytics();

    if (loading && !summary) {
        return (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh' }}>
                <Loader size={40} />
            </div>
        );
    }

    if (error) {
        return <div style={{ color: '#ef4444' }}>{error}</div>;
    }

    const stats = [
        {
            label: 'Total Documents',
            value: summary?.total_documents ?? 0,
            icon: Files,
            color: 'var(--accent-cyan)',
            description: 'Indexed and searchable'
        },
        {
            label: 'Total Queries',
            value: summary?.total_queries ?? 0,
            icon: MessageSquare,
            color: '#8b5cf6',
            description: 'Questions asked by users'
        },
        {
            label: 'Avg. Similarity',
            value: summary?.average_similarity ? `${(summary.average_similarity * 100).toFixed(1)}%` : '0%',
            icon: Target,
            color: '#22c55e',
            description: 'Average retrieval confidence',
            showProgress: true,
            progress: (summary?.average_similarity ?? 0) * 100
        },
    ];

    return (
        <div className="analytics-page fade-in">
            <header style={{ marginBottom: '40px' }}>
                <h2 style={{ fontSize: '32px', fontWeight: '800', marginBottom: '8px' }}>System Analytics</h2>
                <p style={{ color: 'var(--text-secondary)' }}>Insights into your AI assistant's performance and usage.</p>
            </header>

            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                gap: '24px',
                marginBottom: '40px'
            }}>
                {stats.map((stat, index) => (
                    <div key={index} className="glass" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                            <div style={{
                                width: '44px',
                                height: '44px',
                                borderRadius: '12px',
                                background: `${stat.color}15`,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                color: stat.color,
                                border: `1px solid ${stat.color}30`
                            }}>
                                <stat.icon size={22} />
                            </div>
                            <TrendingUp size={16} style={{ color: '#22c55e', opacity: 0.6 }} />
                        </div>

                        <div>
                            <p style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '4px' }}>{stat.label}</p>
                            <h3 style={{ fontSize: '28px', fontWeight: '800' }}>{stat.value}</h3>
                        </div>

                        {stat.showProgress && (
                            <div style={{ width: '100%', height: '6px', background: 'var(--bg-input)', borderRadius: '10px', overflow: 'hidden' }}>
                                <div style={{
                                    width: `${stat.progress}%`,
                                    height: '100%',
                                    background: stat.color,
                                    borderRadius: '10px',
                                    boxShadow: `0 0 10px ${stat.color}40`
                                }} />
                            </div>
                        )}

                        <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{stat.description}</p>
                    </div>
                ))}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '24px' }}>
                {/* Most Queried Document Card */}
                <div className="glass" style={{ padding: '32px', display: 'flex', alignItems: 'center', gap: '32px' }}>
                    <div style={{
                        width: '80px',
                        height: '80px',
                        borderRadius: '20px',
                        background: 'linear-gradient(135deg, rgba(234, 179, 8, 0.1) 0%, rgba(234, 179, 8, 0.05) 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: '#eab308',
                        border: '1px solid rgba(234, 179, 8, 0.2)'
                    }}>
                        <Trophy size={40} />
                    </div>

                    <div style={{ flex: 1 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                            <BarChart size={18} style={{ color: '#eab308' }} />
                            <span style={{ fontSize: '14px', fontWeight: '700', color: '#eab308', letterSpacing: '0.5px' }}>TOP PERFORMING DOCUMENT</span>
                        </div>

                        {summary?.most_queried_document ? (
                            <>
                                <h3 style={{ fontSize: '24px', fontWeight: '800', marginBottom: '4px' }}>
                                    {summary.most_queried_document.filename}
                                </h3>
                                <p style={{ color: 'var(--text-secondary)' }}>
                                    This document has been successfully used as a source for <strong>{summary.most_queried_document.query_count}</strong> queries.
                                </p>
                            </>
                        ) : (
                            <p style={{ color: 'var(--text-secondary)' }}>No query data available yet.</p>
                        )}
                    </div>

                    <div style={{ textAlign: 'right' }}>
                        <span style={{ fontSize: '48px', fontWeight: '900', opacity: 0.1, fontStyle: 'italic' }}>#1</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AnalyticsPage;

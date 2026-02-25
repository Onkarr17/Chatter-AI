import React from 'react';
import { useTheme } from '../context/ThemeContext';
import { MessageSquare, Files, BarChart3, Sun, Moon, Zap } from 'lucide-react';

interface LayoutProps {
    activeTab: 'chat' | 'docs' | 'analytics';
    setActiveTab: (tab: 'chat' | 'docs' | 'analytics') => void;
    children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ activeTab, setActiveTab, children }) => {
    const { theme, toggleTheme } = useTheme();

    const navItems = [
        { id: 'chat', label: 'Chat', icon: MessageSquare },
        { id: 'docs', label: 'Documents', icon: Files },
        { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    ] as const;

    return (
        <div className="layout" style={{ display: 'flex', minHeight: '100vh' }}>
            {/* Sidebar */}
            <aside className="glass" style={{
                width: '280px',
                height: '100vh',
                position: 'fixed',
                left: 0,
                top: 0,
                display: 'flex',
                flexDirection: 'column',
                padding: '24px',
                borderRadius: 0,
                borderLeft: 'none',
                borderTop: 'none',
                borderBottom: 'none',
                zIndex: 100,
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '40px' }}>
                    <div style={{
                        width: '40px',
                        height: '40px',
                        borderRadius: '10px',
                        background: 'var(--accent-gradient)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'white'
                    }}>
                        <Zap size={24} />
                    </div>
                    <h1 style={{ fontSize: '24px', fontWeight: '800', letterSpacing: '-0.5px' }}>Chatter AI</h1>
                </div>

                <nav style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {navItems.map((item) => (
                        <button
                            key={item.id}
                            onClick={() => setActiveTab(item.id)}
                            className={activeTab === item.id ? 'glass-accent' : ''}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '12px',
                                padding: '12px 16px',
                                borderRadius: 'var(--radius-md)',
                                border: activeTab === item.id ? '1px solid var(--border-accent)' : '1px solid transparent',
                                background: activeTab === item.id ? 'rgba(6, 182, 212, 0.1)' : 'transparent',
                                color: activeTab === item.id ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                                cursor: 'pointer',
                                transition: 'all 0.2s ease',
                                textAlign: 'left',
                                fontSize: '16px',
                                fontWeight: activeTab === item.id ? '600' : '400',
                            }}
                        >
                            <item.icon size={20} />
                            {item.label}
                        </button>
                    ))}
                </nav>

                <div style={{ marginTop: 'auto', paddingTop: '20px', borderTop: '1px solid var(--border-color)' }}>
                    <button
                        onClick={toggleTheme}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            width: '100%',
                            padding: '12px 16px',
                            borderRadius: 'var(--radius-md)',
                            border: '1px solid var(--border-color)',
                            background: 'var(--bg-input)',
                            color: 'var(--text-primary)',
                            cursor: 'pointer',
                            fontSize: '14px',
                        }}
                    >
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                            {theme === 'dark' ? <Moon size={18} /> : <Sun size={18} />}
                            {theme === 'dark' ? 'Dark Mode' : 'Light Mode'}
                        </div>
                        <div style={{
                            width: '36px',
                            height: '20px',
                            borderRadius: '20px',
                            background: theme === 'dark' ? 'var(--accent-cyan)' : '#cbd5e1',
                            position: 'relative',
                            transition: 'background 0.3s ease'
                        }}>
                            <div style={{
                                width: '16px',
                                height: '16px',
                                borderRadius: '50%',
                                background: 'white',
                                position: 'absolute',
                                top: '2px',
                                left: theme === 'dark' ? '18px' : '2px',
                                transition: 'left 0.3s ease'
                            }} />
                        </div>
                    </button>
                </div>
            </aside>

            {/* Main Content */}
            <main style={{
                flex: 1,
                marginLeft: '280px',
                padding: '40px',
                maxWidth: '1200px',
                width: '100%',
                animation: 'fadeIn 0.5s ease-out',
            }}>
                {children}
            </main>
        </div>
    );
};

export default Layout;

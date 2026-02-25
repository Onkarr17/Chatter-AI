import React from 'react';
import { useChatContext } from '../context/ChatContext';
import { Send, FileText, User, Bot, Trash2 } from 'lucide-react';
import Loader from '../components/common/Loader';

const ChatPage: React.FC = () => {
    const {
        messages,
        question,
        setQuestion,
        loading,
        error,
        handleAsk,
        selectedFile,
        clearChat,
        hasActiveDocs
    } = useChatContext();

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleAsk();
        }
    };

    return (
        <div className="chat-page fade-in" style={{
            display: 'flex',
            flexDirection: 'column',
            height: 'calc(100vh - 80px)', // Adjusted for Layout padding
            maxWidth: '900px',
            margin: '0 auto'
        }}>
            <header style={{
                marginBottom: '24px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
            }}>
                <div>
                    <h2 style={{ fontSize: '28px', fontWeight: '800' }}>AI Chat</h2>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
                        {selectedFile
                            ? `✨ Searching all PDFs | Recently uploaded: ${selectedFile.name}`
                            : hasActiveDocs
                                ? '✨ Searching across all indexed documents'
                                : 'Upload a PDF to begin conversational search'}
                    </p>
                </div>
                {messages.length > 0 && (
                    <button
                        onClick={clearChat}
                        style={{
                            padding: '8px 12px',
                            borderRadius: 'var(--radius-sm)',
                            background: 'transparent',
                            border: '1px solid var(--border-color)',
                            color: 'var(--text-muted)',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '6px',
                            fontSize: '13px'
                        }}
                    >
                        <Trash2 size={14} /> Clear
                    </button>
                )}
            </header>

            {/* Chat Messages Area */}
            <div style={{
                flex: 1,
                overflowY: 'auto',
                paddingRight: '12px',
                display: 'flex',
                flexDirection: 'column',
                gap: '24px',
                marginBottom: '24px'
            }}>
                {messages.length === 0 ? (
                    <div style={{
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'var(--text-muted)',
                        textAlign: 'center',
                        opacity: 0.6
                    }}>
                        <Bot size={48} style={{ marginBottom: '16px' }} />
                        <p>No messages yet. Ask something about your documents!</p>
                    </div>
                ) : (
                    messages.map((msg, idx) => (
                        <div
                            key={idx}
                            className="fade-in"
                            style={{
                                display: 'flex',
                                gap: '16px',
                                alignItems: 'flex-start',
                                flexDirection: msg.role === 'user' ? 'row-reverse' : 'row'
                            }}
                        >
                            <div style={{
                                width: '36px',
                                height: '36px',
                                borderRadius: '8px',
                                background: msg.role === 'user' ? 'var(--accent-cyan)' : 'var(--bg-input)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                color: msg.role === 'user' ? 'white' : 'var(--accent-cyan)',
                                border: msg.role === 'user' ? 'none' : '1px solid var(--border-color)',
                                flexShrink: 0
                            }}>
                                {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
                            </div>

                            <div style={{
                                maxWidth: '80%',
                                padding: '16px 20px',
                                borderRadius: 'var(--radius-md)',
                                background: msg.role === 'user' ? 'var(--bg-input)' : 'var(--bg-card)',
                                border: '1px solid var(--border-color)',
                                color: 'var(--text-primary)',
                                position: 'relative'
                            }}>
                                <p style={{ fontSize: '15px', lineHeight: '1.6', whiteSpace: 'pre-wrap' }}>{msg.content}</p>
                                {msg.similarity !== undefined && (
                                    <div style={{
                                        marginTop: '12px',
                                        fontSize: '11px',
                                        color: 'var(--accent-cyan)',
                                        fontWeight: '600',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '4px'
                                    }}>
                                        <FileText size={12} />
                                        Confidence: {(msg.similarity * 100).toFixed(0)}%
                                    </div>
                                )}
                            </div>
                        </div>
                    ))
                )}
                {loading && (
                    <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                        <div style={{
                            width: '36px',
                            height: '36px',
                            borderRadius: '8px',
                            background: 'var(--bg-input)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: 'var(--accent-cyan)',
                            border: '1px solid var(--border-color)'
                        }}>
                            <Bot size={20} />
                        </div>
                        <Loader size={18} />
                    </div>
                )}
            </div>

            {/* Input Section */}
            <div style={{ position: 'relative', paddingBottom: '20px' }}>
                {error && (
                    <div style={{
                        position: 'absolute',
                        top: '-40px',
                        left: 0,
                        right: 0,
                        color: '#ef4444',
                        fontSize: '13px',
                        textAlign: 'center'
                    }}>
                        {error}
                    </div>
                )}
                <div style={{ position: 'relative' }}>
                    <textarea
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder={selectedFile || hasActiveDocs ? "Ask a question..." : "Please upload a document first"}
                        disabled={(!selectedFile && !hasActiveDocs) || loading}
                        style={{
                            width: '100%',
                            minHeight: '60px',
                            maxHeight: '200px',
                            padding: '18px 60px 18px 24px',
                            borderRadius: 'var(--radius-lg)',
                            background: 'var(--bg-input)',
                            border: '1px solid var(--border-color)',
                            color: 'var(--text-primary)',
                            fontSize: '16px',
                            resize: 'none',
                            outline: 'none',
                            fontFamily: 'inherit',
                            transition: 'all 0.3s ease',
                            boxShadow: 'var(--shadow-md)'
                        }}
                    />
                    <button
                        onClick={handleAsk}
                        disabled={loading || (!selectedFile && !hasActiveDocs) || !question.trim()}
                        style={{
                            position: 'absolute',
                            right: '12px',
                            top: '50%',
                            transform: 'translateY(-50%)',
                            width: '40px',
                            height: '40px',
                            borderRadius: '10px',
                            background: 'var(--accent-cyan)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: 'white',
                            border: 'none',
                            cursor: 'pointer',
                            opacity: (loading || (!selectedFile && !hasActiveDocs) || !question.trim()) ? 0.5 : 1,
                            transition: 'all 0.2s ease',
                        }}
                    >
                        <Send size={20} />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ChatPage;

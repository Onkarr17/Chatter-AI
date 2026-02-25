import React, { useRef, useState } from 'react';
import { useDocuments } from '../hooks/useDocuments';
import { useChatContext } from '../context/ChatContext';
import { chatApi } from '../api/chat.api';
import { FileText, X, Clock, File, Upload, CheckCircle2 } from 'lucide-react';
import Loader from '../components/common/Loader';

const DocumentsPage: React.FC = () => {
    const { documents, loading, refresh, removeDocument } = useDocuments();
    const { selectedFile, setSelectedFile } = useChatContext();
    const fileInputRef = useRef<HTMLInputElement>(null);

    const [uploading, setUploading] = useState(false);
    const [uploadError, setUploadError] = useState<string | null>(null);

    const onFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setSelectedFile(file);
            setUploadError(null);

            // Auto-trigger upload/index
            setUploading(true);
            try {
                // Using a generic question to trigger backend indexing
                await chatApi.submitQuery(file, "Process and index this document.");
                await refresh(); // Refresh the list to show the new document

                // Reset input value to allow re-selection of the same file if needed
                if (fileInputRef.current) {
                    fileInputRef.current.value = '';
                }
            } catch (err: any) {
                console.error('Upload failed:', err);
                const errorMsg = err.response?.data?.detail || 'Failed to index document. Backend might be down.';
                setUploadError(errorMsg);
            } finally {
                setUploading(false);
            }
        }
    };

    return (
        <div className="documents-page fade-in">
            <header style={{ marginBottom: '40px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
                <div>
                    <h2 style={{ fontSize: '32px', fontWeight: '800', marginBottom: '8px' }}>Document Hub</h2>
                    <p style={{ color: 'var(--text-secondary)' }}>Upload and manage PDFs for the AI Assistant.</p>
                </div>
                <button
                    onClick={refresh}
                    style={{
                        padding: '10px 20px',
                        borderRadius: 'var(--radius-md)',
                        background: 'var(--bg-card)',
                        border: '1px solid var(--border-color)',
                        color: 'var(--text-primary)',
                        cursor: 'pointer',
                        fontSize: '14px',
                        fontWeight: '600'
                    }}
                >
                    Refresh List
                </button>
            </header>
            {/* Upload Feedback */}
            {uploadError && (
                <div style={{
                    background: 'rgba(239, 68, 68, 0.1)',
                    border: '1px solid rgba(239, 68, 68, 0.2)',
                    color: '#ef4444',
                    padding: '12px',
                    borderRadius: 'var(--radius-md)',
                    marginBottom: '20px',
                    fontSize: '14px',
                    textAlign: 'center'
                }}>
                    ⚠️ {uploadError}
                </div>
            )}

            <div
                className="glass"
                onClick={() => !uploading && fileInputRef.current?.click()}
                style={{
                    padding: '40px',
                    textAlign: 'center',
                    cursor: uploading ? 'default' : 'pointer',
                    borderStyle: 'dashed',
                    borderColor: uploading ? 'var(--accent-cyan)' : selectedFile ? 'var(--accent-cyan)' : 'var(--border-color)',
                    background: selectedFile || uploading ? 'rgba(6, 182, 212, 0.05)' : 'var(--bg-card)',
                    transition: 'all 0.3s ease',
                    marginBottom: '40px',
                    opacity: uploading ? 0.7 : 1
                }}
            >
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={onFileChange}
                    accept="application/pdf"
                    style={{ display: 'none' }}
                />
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '12px'
                }}>
                    <div style={{
                        width: '56px',
                        height: '56px',
                        borderRadius: '50%',
                        background: selectedFile || uploading ? 'var(--accent-cyan)' : 'var(--bg-input)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: selectedFile || uploading ? 'white' : 'var(--text-muted)',
                        marginBottom: '8px'
                    }}>
                        {uploading ? <Loader size={32} color="white" /> : selectedFile ? <CheckCircle2 size={32} /> : <Upload size={32} />}
                    </div>
                    {uploading ? (
                        <div>
                            <p style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>Indexing...</p>
                            <p style={{ fontSize: '14px', color: 'var(--accent-cyan)', marginTop: '4px' }}>Please wait while we process your PDF</p>
                        </div>
                    ) : selectedFile ? (
                        <div>
                            <p style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>{selectedFile.name}</p>
                            <p style={{ fontSize: '14px', color: 'var(--accent-cyan)', marginTop: '4px' }}>Ready for Chat (Click to change)</p>
                        </div>
                    ) : (
                        <div>
                            <p style={{ fontSize: '18px', fontWeight: '700', color: 'var(--text-primary)' }}>Upload Document</p>
                            <p style={{ fontSize: '14px', color: 'var(--text-secondary)', marginTop: '4px' }}>Select a PDF to start analyzing</p>
                        </div>
                    )}
                </div>
            </div>

            <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '40px' }}>
                <h3 style={{ fontSize: '20px', fontWeight: '700', marginBottom: '24px' }}>Indexed Documents</h3>

                {loading && documents.length === 0 ? (
                    <div style={{ display: 'flex', justifyContent: 'center', py: 8 }}>
                        <Loader size={40} />
                    </div>
                ) : documents.length === 0 ? (
                    <div className="glass" style={{ padding: '60px', textAlign: 'center', borderStyle: 'none' }}>
                        <File size={40} style={{ color: 'var(--text-muted)', marginBottom: '16px' }} />
                        <p style={{ color: 'var(--text-secondary)' }}>No indexed documents found on the server.</p>
                    </div>
                ) : (
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
                        gap: '24px',
                    }}>
                        {documents.map((doc) => (
                            <div
                                key={doc.document_id}
                                className="glass fade-in"
                                style={{
                                    position: 'relative',
                                    display: 'flex',
                                    flexDirection: 'column',
                                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                                }}
                            >
                                <div style={{
                                    height: '140px',
                                    background: 'var(--bg-input)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    color: 'var(--accent-cyan)',
                                    borderRadius: 'calc(var(--radius-md) - 1px) calc(var(--radius-md) - 1px) 0 0',
                                    overflow: 'hidden',
                                    position: 'relative',
                                    borderBottom: '1px solid var(--border-color)'
                                }}>
                                    <img
                                        src={`/api/documents/thumbnail/${doc.document_id}`}
                                        alt={doc.filename}
                                        style={{
                                            width: '100%',
                                            height: '100%',
                                            objectFit: 'cover',
                                            opacity: 0.8,
                                            transition: 'opacity 0.3s ease'
                                        }}
                                        onLoad={(e) => (e.currentTarget.style.opacity = '1')}
                                        onError={(e) => {
                                            e.currentTarget.style.display = 'none';
                                            const fallback = e.currentTarget.parentElement?.querySelector('.fallback-icon');
                                            if (fallback) (fallback as HTMLElement).style.display = 'flex';
                                        }}
                                    />
                                    <div className="fallback-icon" style={{
                                        display: 'none',
                                        position: 'absolute',
                                        inset: 0,
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        background: 'linear-gradient(45deg, #1e293b, #0f172a)'
                                    }}>
                                        <FileText size={40} />
                                    </div>
                                </div>

                                <button
                                    onClick={() => removeDocument(doc.document_id)}
                                    className="delete-btn"
                                    style={{
                                        position: 'absolute',
                                        top: '12px',
                                        right: '12px',
                                        height: '32px',
                                        borderRadius: 'var(--radius-full)',
                                        background: 'rgba(0,0,0,0.5)',
                                        border: 'none',
                                        color: 'white',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        cursor: 'pointer',
                                        backdropFilter: 'blur(4px)',
                                        transition: 'all 0.3s ease',
                                        padding: '0 8px',
                                        overflow: 'hidden',
                                        width: 'auto'
                                    }}
                                >
                                    <X size={16} />
                                    <span className="remove-text" style={{
                                        maxWidth: 0,
                                        opacity: 0,
                                        fontSize: '13px',
                                        fontWeight: '600',
                                        marginLeft: 0,
                                        transition: 'all 0.3s ease',
                                        whiteSpace: 'nowrap'
                                    }}>Remove</span>
                                </button>

                                <div style={{ padding: '20px' }}>
                                    <h4 style={{ fontSize: '16px', fontWeight: '700', marginBottom: '12px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        {doc.filename}
                                    </h4>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--text-secondary)' }}>
                                            <FileText size={14} />
                                            <span>{doc.total_pages} Pages</span>
                                        </div>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--text-secondary)' }}>
                                            <Clock size={14} />
                                            <span>{new Date(doc.created_at).toLocaleDateString()}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default DocumentsPage;

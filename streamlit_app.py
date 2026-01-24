import os
import time
import streamlit as st

# Import RAG backend functions
# Note: step1_read_pdf is imported but not used - we use step1_read_pdf_custom instead
# PDF_PATH in rag.py is only for standalone use, not used by Streamlit
from rag import (
    step2_split_into_chunks, step3_create_embeddings,
    step4_build_faiss_index, step4_retrieve_top_chunks,
    step5_answer_from_retrieved, step5_general_knowledge_answer,
    step6_get_doc_hash_from_path, step6_cache_paths,
    step6_load_cache, step6_save_cache, CACHE_VERSION
)

# ------------------------------------------------------------
# UI CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="ChatterAI",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
/* Remove top padding and constrain width for better readability */
.block-container { 
    padding-top: 1.2rem;
    max-width: 980px !important;
}

/* Hide Streamlit's default menu and deploy button */
#MainMenu { visibility: hidden; }
header { visibility: hidden; }
footer { visibility: hidden; }
div[data-testid="stToolbar"] { visibility: hidden; }
div[data-testid="stDecoration"] { visibility: hidden; }
button[title="View app source"] { display: none; }
button[title="Deploy this app"] { display: none; }

/* Hide and disable default Streamlit sidebar toggle button */
button[data-testid="baseButton-header"],
button[data-testid="baseButton-header"]:hover,
button[data-testid="baseButton-header"]:active,
button[data-testid="baseButton-header"]:focus,
section[data-testid="stSidebar"] ~ * button[data-testid="baseButton-header"],
.stApp button[data-testid="baseButton-header"],
header button[data-testid="baseButton-header"],
[data-testid="stHeader"] button[data-testid="baseButton-header"],
button[aria-label*="sidebar"],
button[aria-label*="menu"],
button[title*="sidebar"],
button[title*="menu"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
}

/* Force sidebar to always be visible */
section[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    transform: translateX(0) !important;
    opacity: 1 !important;
}

.stApp.sidebar-collapsed section[data-testid="stSidebar"],
section[data-testid="stSidebar"].collapsed {
    display: block !important;
    visibility: visible !important;
    transform: translateX(0) !important;
    opacity: 1 !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(20,20,25,0.95), rgba(10,10,15,0.98));
    border-right: 1px solid rgba(255,255,255,0.06);
    padding-top: 0 !important;
    min-width: 21.125rem !important;
    overflow-y: visible !important;
    overflow-x: visible !important;
    height: auto !important;
    max-height: 100vh !important;
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}

section[data-testid="stSidebar"] > div {
    overflow-y: visible !important;
    overflow-x: visible !important;
    height: auto !important;
    max-height: none !important;
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}

/* Remove gap above sidebar content */
section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Main background subtle gradient with cyan accent */
.stApp {
    background: radial-gradient(circle at top, rgba(80,80,120,0.18), rgba(0,0,0,0) 45%),
                radial-gradient(circle at bottom, rgba(6,182,212,0.15), rgba(0,0,0,0) 40%),
                #0b0f15;
}

/* Title styling */
h1 {
    font-weight: 800 !important;
    letter-spacing: -1px;
}

/* Sticky header for product name with cyan accent */
.sticky-header-container {
    position: sticky !important;
    top: 0 !important;
    z-index: 999 !important;
    background: rgba(11,15,21,0.95) !important;
    padding: 1rem 0 !important;
    margin-bottom: 1rem !important;
    border-bottom: 2px solid rgba(6,182,212,0.3) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    backdrop-filter: blur(12px) !important;
}

.sticky-header-container h1 {
    margin-top: 0 !important;
}

/* Upload box => glass card with cyan accent */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(6,182,212,0.2);
    padding: 16px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    transition: border-color 0.3s ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(6,182,212,0.4);
}

/* Chat input => keep original size, remove circular button */
div[data-testid="stChatInput"] {
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
}

div[data-testid="stChatInput"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="stChatInput"] textarea {
    border-radius: 18px !important;
    border: 1px solid rgba(6,182,212,0.3) !important;
    background: rgba(255,255,255,0.04) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    padding: 14px 18px !important;
    font-size: 1rem !important;
    min-height: 56px !important;
}

div[data-testid="stChatInput"] textarea:focus {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 0 2px rgba(6,182,212,0.2) !important;
    outline: none !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255,255,255,0.5) !important;
}

/* Style the send button to be inside the input on the right with solid color */
div[data-testid="stChatInput"] > div {
    position: relative !important;
}

div[data-testid="stChatInput"] button {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    position: absolute !important;
    right: 32px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: 40px !important;
    height: 40px !important;
    border-radius: 10px !important;
    background: #06b6d4 !important;
    border: none !important;
    color: white !important;
    padding: 0 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    z-index: 10 !important;
    box-shadow: 0 2px 4px rgba(6,182,212,0.3) !important;
}

div[data-testid="stChatInput"] button:hover {
    background: #0891b2 !important;
    transform: translateY(-50%) scale(1.05) !important;
    box-shadow: 0 4px 8px rgba(6,182,212,0.4) !important;
}

div[data-testid="stChatInput"] button:active {
    transform: translateY(-50%) scale(0.95) !important;
}

div[data-testid="stChatInput"] textarea {
    padding-right: 54px !important;
}

/* Buttons with cyan accent */
.stButton button {
    border-radius: 14px;
    border: 1px solid rgba(6,182,212,0.3) !important;
    background: rgba(255,255,255,0.03);
    color: rgba(255,255,255,0.9) !important;
    transition: all 0.3s ease;
}
.stButton button:hover {
    border-color: #06b6d4 !important;
    background: rgba(6,182,212,0.1) !important;
    box-shadow: 0 0 0 2px rgba(6,182,212,0.2) !important;
    transform: translateY(-1px);
}
.stButton button[kind="primary"] {
    background: #06b6d4 !important;
    border-color: #06b6d4 !important;
    color: white !important;
}
.stButton button[kind="primary"]:hover {
    background: #0891b2 !important;
    border-color: #0891b2 !important;
    box-shadow: 0 4px 12px rgba(6,182,212,0.4) !important;
}

/* Radio buttons with cyan accent */
div[role="radiogroup"] label {
    color: rgba(255,255,255,0.8) !important;
}
div[role="radiogroup"] label span:first-child {
    color: #06b6d4 !important;
}
div[role="radiogroup"] input[type="radio"]:checked + label span:first-child {
    color: #06b6d4 !important;
    font-weight: 600;
}
div[role="radiogroup"] input[type="radio"]:checked {
    accent-color: #06b6d4 !important;
}

/* Sliders with cyan accent */
.stSlider > div > div > div {
    background: rgba(6,182,212,0.2) !important;
}
.stSlider > div > div > div > div {
    background: #06b6d4 !important;
}
.stSlider label {
    color: rgba(255,255,255,0.8) !important;
}

/* Expander styling with cyan accent */
details {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(6,182,212,0.2);
    border-radius: 16px;
    padding: 10px;
    transition: border-color 0.3s ease;
}
details:hover {
    border-color: rgba(6,182,212,0.4);
}
details summary {
    color: rgba(255,255,255,0.9) !important;
}
details[open] {
    border-color: rgba(6,182,212,0.5);
}

/* Remove Streamlit "focus" outline look */
:focus { outline: none !important; }

/* Links and highlights with cyan accent */
a {
    color: #06b6d4 !important;
}
a:hover {
    color: #22d3ee !important;
}

/* Success/Info messages with cyan accent */
.stSuccess {
    border-left: 4px solid #06b6d4 !important;
}
.stInfo {
    border-left: 4px solid #06b6d4 !important;
}

/* File uploader drag state */
div[data-testid="stFileUploader"]:has(input:focus) {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 0 2px rgba(6,182,212,0.2) !important;
}

/* Hide the default "200MB" limit text from Streamlit file uploader */
/* Target only the paragraph that comes after the file input (contains "Limit 200MB per file") */
div[data-testid="stFileUploader"] > div > p:not([class*="help"]),
div[data-testid="stFileUploader"] > div > small {
    display: none !important;
    visibility: hidden !important;
}

/* More specific: Hide paragraphs that are direct children of the uploader container but not help text */
div[data-testid="stFileUploader"] > p,
div[data-testid="stFileUploader"] > small {
    display: none !important;
}

/* Keep the label and help text visible */
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] [data-testid="stTooltipIcon"],
div[data-testid="stFileUploader"] [class*="help"] {
    display: block !important;
}

/* More aggressive hiding for file uploader limit text */
div[data-testid="stFileUploader"] p:has-text("200MB"),
div[data-testid="stFileUploader"] p:has-text("Limit") {
    display: none !important;
}

.chat-bubble {
    padding: 12px 16px;
    border-radius: 14px;
    margin-bottom: 10px;
    line-height: 1.5;
}
.user-bubble {
    background: rgba(6,182,212,0.15);
    color: white;
    border: 1px solid rgba(6,182,212,0.4) !important;
}
.assistant-bubble {
    background: rgba(255,255,255,0.05);
    color: rgba(255,255,255,0.9);
    border: 1px solid rgba(255,255,255,0.1);
}

/* Style Streamlit chat message bubbles - ChatGPT style layout */
div[data-testid="stChatMessage"] {
    margin-bottom: 1.5rem;
    display: flex !important;
    width: 100% !important;
    position: relative;
}

/* Subtle divider between responses */
div[data-testid="stChatMessage"]:not(:last-child)::after {
    content: '';
    position: absolute;
    bottom: -0.75rem;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(
        to right,
        transparent,
        rgba(255,255,255,0.08) 20%,
        rgba(255,255,255,0.08) 80%,
        transparent
    );
}

/* User message bubble - left side with accent border */
div[data-testid="stChatMessage"][data-message-author="user"] {
    justify-content: flex-start !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    max-width: 75% !important;
}

div[data-testid="stChatMessage"][data-message-author="user"] > div {
    background: rgba(6,182,212,0.15) !important;
    border: 1px solid rgba(6,182,212,0.5) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 12px 16px !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    color: rgba(255,255,255,0.95) !important;
}

/* Assistant message bubble - right side with neutral styling */
div[data-testid="stChatMessage"][data-message-author="assistant"] {
    justify-content: flex-end !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    max-width: 75% !important;
}

div[data-testid="stChatMessage"][data-message-author="assistant"] > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 16px !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    color: rgba(255,255,255,0.9) !important;
}

/* Hide the avatar icons for cleaner look */
div[data-testid="stChatMessage"] [data-testid="stChatAvatar"] {
    display: none !important;
}
.small-note {
    font-size: 12px;
    opacity: 0.75;
}
</style>
<script>
// Smooth scrolling to latest message (premium feel)
function scrollToLatestMessage() {
    const chatMessages = document.querySelectorAll('[data-testid="stChatMessage"]');
    if (chatMessages.length > 0) {
        const lastMessage = chatMessages[chatMessages.length - 1];
        lastMessage.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'end',
            inline: 'nearest'
        });
    }
}

// Scroll on page load
window.addEventListener('load', function() {
    setTimeout(scrollToLatestMessage, 300);
});

// Watch for new messages and scroll smoothly
const chatObserver = new MutationObserver(function(mutations) {
    let shouldScroll = false;
    mutations.forEach(function(mutation) {
        if (mutation.addedNodes.length > 0) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === 1 && (
                    node.hasAttribute && node.hasAttribute('data-testid') && 
                    node.getAttribute('data-testid') === 'stChatMessage'
                )) {
                    shouldScroll = true;
                }
            });
        }
    });
    
    if (shouldScroll) {
        setTimeout(scrollToLatestMessage, 100);
    }
});

// Observe the main content area for new messages
const mainContent = document.querySelector('main') || document.body;
if (mainContent) {
    chatObserver.observe(mainContent, {
        childList: true,
        subtree: true
    });
}

// Also scroll after Streamlit reruns
setTimeout(scrollToLatestMessage, 500);
setTimeout(scrollToLatestMessage, 1000);

// Disable default Streamlit sidebar toggle button and force sidebar visible
(function() {
    function disableSidebarToggle() {
        // Find and remove/hide the toggle button
        const toggleButtons = document.querySelectorAll(
            'button[data-testid="baseButton-header"], ' +
            'button[aria-label*="sidebar"], ' +
            'button[aria-label*="menu"], ' +
            'button[title*="sidebar"], ' +
            'button[title*="menu"]'
        );
        
        toggleButtons.forEach(function(btn) {
            // Remove from DOM completely
            try {
                btn.remove();
            } catch(e) {
                // If remove fails, hide it aggressively
                btn.style.display = 'none';
                btn.style.visibility = 'hidden';
                btn.style.opacity = '0';
                btn.style.pointerEvents = 'none';
                btn.style.width = '0';
                btn.style.height = '0';
                btn.style.position = 'absolute';
                btn.style.left = '-9999px';
                // Prevent clicks
                btn.onclick = function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    return false;
                };
            }
        });
        
        // Force sidebar to be visible
        const sidebar = document.querySelector('section[data-testid="stSidebar"]');
        const app = document.querySelector('.stApp');
        
        if (sidebar) {
            // Remove collapsed classes
            if (app) {
                app.classList.remove('sidebar-collapsed');
            }
            sidebar.classList.remove('collapsed');
            
            // Force visible
            sidebar.style.display = 'block';
            sidebar.style.visibility = 'visible';
            sidebar.style.transform = 'translateX(0)';
            sidebar.style.opacity = '1';
        }
        
        // Clear sidebar state from storage
        try {
            const keys = Object.keys(localStorage);
            keys.forEach(function(key) {
                if (key.includes('sidebar') || key.includes('collapsed')) {
                    localStorage.removeItem(key);
                }
            });
            
            const sessionKeys = Object.keys(sessionStorage);
            sessionKeys.forEach(function(key) {
                if (key.includes('sidebar') || key.includes('collapsed')) {
                    sessionStorage.removeItem(key);
                }
            });
        } catch(e) {
            // Ignore storage errors
        }
    }
    
    // Run immediately
    disableSidebarToggle();
    
    // Run on page load
    window.addEventListener('load', disableSidebarToggle);
    
    // Run after delays to catch Streamlit reruns
    setTimeout(disableSidebarToggle, 100);
    setTimeout(disableSidebarToggle, 500);
    setTimeout(disableSidebarToggle, 1000);
    
    // Watch for DOM changes and disable toggle button
    const observer = new MutationObserver(function() {
        disableSidebarToggle();
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['class', 'style']
    });
    
    // Also run periodically to catch any buttons that appear
    setInterval(disableSidebarToggle, 500);
})();
</script>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# SESSION STATE INIT
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "messages_llm" not in st.session_state:
    # Backward-compatible migration: keep any existing messages in LLM history
    st.session_state.messages_llm = list(st.session_state.messages)

if "messages_pdf_dict" not in st.session_state:
    st.session_state.messages_pdf_dict = {}

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "embedder" not in st.session_state:
    st.session_state.embedder = None

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

if "doc_hash" not in st.session_state:
    st.session_state.doc_hash = None

if "prefill" not in st.session_state:
    st.session_state.prefill = ""


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
# Product name - large and prominent
st.sidebar.markdown("""
<h1 style="font-size: 28px; font-weight: bold; margin-bottom: 10px; margin-top: 0; padding-top: 0;">🤖 Chatter AI</h1>
""", unsafe_allow_html=True)
st.sidebar.divider()

# 1. Mode selection (First)
st.sidebar.subheader("🧠 Mode")
mode = st.sidebar.radio(
    "Select mode", 
    ["📄 PDF RAG", "🌍 LLM Chat"], 
    index=1,  # Default to LLM Chat
    label_visibility="collapsed"
)

st.sidebar.divider()

# 3. Answer Length presets (Third)
st.sidebar.subheader("📊 Answer Length")

# Preset configurations
presets = {
    "Small": {
        "top_k": 3,
        "threshold": 0.25,
        "chunk_size": 600,
        "chunk_overlap": 100,
        "max_tokens": 150,
        "description": "Fast, concise answers (2-3 sentences)"
    },
    "Medium": {
        "top_k": 6,
        "threshold": 0.30,
        "chunk_size": 800,
        "chunk_overlap": 150,
        "max_tokens": 350,
        "description": "Balanced speed and detail (paragraph)"
    },
    "Large": {
        "top_k": 10,
        "threshold": 0.35,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "max_tokens": 700,
        "description": "Detailed, comprehensive answers (multiple paragraphs)"
    }
}

# Initialize session state for preset if not exists
if "answer_preset" not in st.session_state:
    st.session_state.answer_preset = "Medium"

# Preset selection buttons
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("Small", use_container_width=True, type="primary" if st.session_state.answer_preset == "Small" else "secondary"):
        st.session_state.answer_preset = "Small"
        st.rerun()
with col2:
    if st.button("Medium", use_container_width=True, type="primary" if st.session_state.answer_preset == "Medium" else "secondary"):
        st.session_state.answer_preset = "Medium"
        st.rerun()
with col3:
    if st.button("Large", use_container_width=True, type="primary" if st.session_state.answer_preset == "Large" else "secondary"):
        st.session_state.answer_preset = "Large"
        st.rerun()

# Get current preset values
current_preset = presets[st.session_state.answer_preset]
top_k = current_preset["top_k"]
threshold = current_preset["threshold"]
chunk_size = current_preset["chunk_size"]
chunk_overlap = current_preset["chunk_overlap"]
max_tokens = current_preset["max_tokens"]

# Show current preset info
st.sidebar.caption(f"📊 {current_preset['description']}")
st.sidebar.caption(f"Chunks: {top_k} | Size: {chunk_size} | Overlap: {chunk_overlap}")

st.sidebar.divider()

# 4. Cache and Actions (Fourth - Last)
st.sidebar.subheader("🔧 Cache & Actions")
colA, colB = st.sidebar.columns(2)
clear_chat = colA.button("🧹 Reset Chat", use_container_width=True)
clear_cache = colB.button("🗑️ Clear Cache", use_container_width=True)

if clear_chat:
    if mode == "🌍 LLM Chat":
        st.session_state.messages_llm = []
    else:
        # Clear only the active PDF's chat, keep other PDFs intact
        current_doc_hash = st.session_state.doc_hash
        if current_doc_hash:
            st.session_state.messages_pdf_dict[current_doc_hash] = []
    st.toast("Chat reset ✅")

if clear_cache:
    # optional: remove entire rag_cache
    try:
        import shutil
        if os.path.exists("rag_cache"):
            shutil.rmtree("rag_cache")
        st.session_state.index = None
        st.session_state.chunks = None
        st.session_state.embedder = None
        st.toast("Cache cleared ✅")
    except Exception as e:
        st.sidebar.error(f"Failed to clear cache: {e}")


# ------------------------------------------------------------
# MAIN UI
# ------------------------------------------------------------
# Sticky header with product name
st.markdown("""
<div id="chatter-ai-header" class="sticky-header-container">
""", unsafe_allow_html=True)
# Dynamic title and caption based on mode
if mode == "🌍 LLM Chat":
    st.title("🤖 Chatter AI — LLM Chat")
    st.caption("Ask any question. Powered by Groq AI.")
    # Status chip for LLM Chat
    st.markdown("""
    <div style="display:flex; gap:8px; align-items:center; margin-top:8px; margin-bottom:12px;">
        <span style="padding:6px 12px; border-radius:999px; background:rgba(34,197,94,0.15);
                     border:1px solid rgba(34,197,94,0.35); font-size:12px; color: #22c55e; white-space:nowrap;">
            ⚡ Groq Connected
        </span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("🤖 Chatter AI — PDF RAG Notebook")
    st.caption("Ask questions from your PDF. Shows sources + similarity scores. Powered by Groq + FAISS.")
    # Status chips for PDF RAG mode
    cache_ready = st.session_state.index is not None
    pdf_loaded = st.session_state.pdf_path is not None and os.path.exists(st.session_state.pdf_path)
    st.markdown(f"""
    <div style="display:flex; gap:8px; align-items:center; margin-top:8px; margin-bottom:12px;">
        <span style="padding:6px 12px; border-radius:999px; background:rgba(6,182,212,0.15);
                     border:1px solid rgba(6,182,212,0.35); font-size:12px; color: #06b6d4; white-space:nowrap;">
            {'⚡ Cache Ready' if cache_ready else '⚡ Cache Not Ready'}
        </span>
        <span style="padding:6px 12px; border-radius:999px; background:rgba(34,197,94,0.15);
                     border:1px solid rgba(34,197,94,0.35); font-size:12px; color: #22c55e; white-space:nowrap;">
            {'📄 PDF Loaded' if pdf_loaded else '📄 PDF Not Loaded'}
        </span>
        <span style="padding:6px 12px; border-radius:999px; background:rgba(34,197,94,0.15);
                     border:1px solid rgba(34,197,94,0.35); font-size:12px; color: #22c55e; white-space:nowrap;">
            ⚡ Groq Connected
        </span>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
</div>
<script>
(function() {
    function makeHeaderSticky() {
        const header = document.getElementById('chatter-ai-header');
        if (header) {
            // Find the Streamlit vertical block container that contains the header
            let container = header.closest('[data-testid="stVerticalBlock"]');
            if (!container) {
                container = header.parentElement;
                while (container && container.tagName !== 'BODY') {
                    if (container.querySelector('h1')) {
                        break;
                    }
                    container = container.parentElement;
                }
            }
            if (container) {
                container.style.position = 'sticky';
                container.style.top = '0';
                container.style.zIndex = '999';
                container.style.backgroundColor = 'white';
                container.style.paddingTop = '1rem';
                container.style.paddingBottom = '1rem';
                container.style.marginBottom = '1rem';
                container.style.borderBottom = '2px solid #e5e7eb';
                container.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
            }
        }
    }
    // Run immediately
    makeHeaderSticky();
    // Also run after a short delay to catch Streamlit's dynamic rendering
    setTimeout(makeHeaderSticky, 100);
    setTimeout(makeHeaderSticky, 500);
})();
</script>
""", unsafe_allow_html=True)

# Compact PDF Upload - will be placed inline with chat input
status_box = st.empty()


# ------------------------------------------------------------
# STREAMING HELPER FUNCTION
# ------------------------------------------------------------
def stream_text(text: str, speed: float = 0.03):
    """
    Streams text with typing effect (word by word).
    Returns a generator that yields text progressively.
    """
    words = text.split(' ')
    current_text = ""
    for i, word in enumerate(words):
        current_text += word
        if i < len(words) - 1:  # Add space except for last word
            current_text += " "
        yield current_text
        time.sleep(speed)  # Delay between words

# ------------------------------------------------------------
# PDF ANSWER FORMATTING
# ------------------------------------------------------------
def format_pdf_answer(text: str) -> str:
    """
    Add extra spacing to make short PDF RAG answers more readable.
    Keeps content intact, only adjusts line breaks.
    """
    if not text:
        return text

    normalized = text.replace("\r\n", "\n").strip()

    # If the answer is short and has no line breaks, add spacing between sentences.
    if "\n" not in normalized and len(normalized) < 600:
        for sep in [". ", "? ", "! "]:
            normalized = normalized.replace(sep, sep.strip() + "\n\n")
        return normalized

    # For multi-line answers, ensure a blank line between list items/paragraphs.
    lines = [ln.strip() for ln in normalized.split("\n") if ln.strip() != ""]
    spaced = []
    for ln in lines:
        spaced.append(ln)
        spaced.append("")  # blank line for spacing
    return "\n".join(spaced).strip()

# ------------------------------------------------------------
# PDF LOAD + BUILD/LOAD INDEX
# ------------------------------------------------------------
def build_or_load_index(pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 150):
    """
    Loads from cache if possible, else builds and saves.
    """
    # Create a custom PDF loader function that uses the uploaded file
    from langchain_community.document_loaders import PyPDFLoader
    
    doc_hash = step6_get_doc_hash_from_path(pdf_path)
    idx_path, meta_path = step6_cache_paths(doc_hash)

    # Always need embedder for question embedding
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if os.path.exists(idx_path) and os.path.exists(meta_path):
        # Load from cache
        index, chunks = step6_load_cache(idx_path, meta_path)
        # Check if old L2 index (shouldn't happen with versioning, but safety check)
        index_type = type(index).__name__
        if "L2" in index_type or "FlatL2" in index_type:
            # Rebuild with cosine similarity
            pages = step1_read_pdf_custom(pdf_path)
            chunks = step2_split_into_chunks(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            embedder, embeddings = step3_create_embeddings(chunks)
            index = step4_build_faiss_index(embeddings)
            step6_save_cache(index, chunks, idx_path, meta_path)
    else:
        # Build new index
        pages = step1_read_pdf_custom(pdf_path)
        chunks = step2_split_into_chunks(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embedder, embeddings = step3_create_embeddings(chunks)
        index = step4_build_faiss_index(embeddings)
        step6_save_cache(index, chunks, idx_path, meta_path)

    return doc_hash, index, chunks, embedder


def step1_read_pdf_custom(pdf_path: str):
    """Custom PDF loader that accepts a path parameter with fallback for malformed PDFs"""
    from langchain_community.document_loaders import PyPDFLoader
    from pypdf import PdfReader
    from langchain_core.documents import Document
    
    try:
        # Try standard PyPDFLoader first
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        if not pages or len(pages) == 0:
            raise ValueError("PDF appears to be empty or could not be read")
        return pages
    except (KeyError, Exception) as e:
        # If standard loader fails (e.g., 'bbox' KeyError), try fallback method
        if 'bbox' in str(e) or 'font' in str(e).lower() or 'KeyError' in str(type(e).__name__):
            try:
                # Fallback: Use pypdf directly with error handling per page
                reader = PdfReader(pdf_path)
                pages = []
                for i, page in enumerate(reader.pages):
                    try:
                        # Try different extraction methods
                        text = None
                        # Method 1: Try with layout mode (if available in pypdf version)
                        try:
                            if hasattr(page, 'extract_text'):
                                # Try with extraction_mode parameter if available
                                import inspect
                                sig = inspect.signature(page.extract_text)
                                if 'extraction_mode' in sig.parameters:
                                    text = page.extract_text(extraction_mode="layout")
                                else:
                                    text = page.extract_text()
                        except Exception:
                            pass
                        
                        # Method 2: If layout mode failed, try plain extraction
                        if not text or text.strip() == "":
                            text = page.extract_text()
                        
                        if text and text.strip():
                            pages.append(Document(
                                page_content=text,
                                metadata={"page": i + 1, "source": pdf_path}
                            ))
                    except Exception as page_error:
                        # Skip problematic pages but continue with others
                        # Only show warning if we're in Streamlit context
                        try:
                            st.warning(f"⚠️ Skipped page {i + 1} due to extraction error. Continuing with other pages...")
                        except:
                            pass  # Not in Streamlit context, skip warning
                        continue
                
                if not pages or len(pages) == 0:
                    raise ValueError("PDF appears to be empty or could not be read with fallback method")
                return pages
            except Exception as fallback_error:
                raise Exception(f"Failed to load PDF from {pdf_path} with both standard and fallback methods. Original error: {str(e)[:200]}. Fallback error: {str(fallback_error)[:200]}")
        else:
            # Re-raise if it's not a font/bbox related error
            raise Exception(f"Failed to load PDF from {pdf_path}: {str(e)}")


# ------------------------------------------------------------
# INTERACTIVE PROMPT SUGGESTIONS (only show in LLM Chat mode, and only if no messages yet)
# ------------------------------------------------------------
if mode == "🌍 LLM Chat" and len(st.session_state.messages_llm) == 0:
    st.markdown("### 💡 Try asking")
    cols = st.columns(3)
    prompts = [
        "What is AGI ?",
        "Any Movie recommendations",
        "Explain Newton's Laws"
    ]
    
    for i, p in enumerate(prompts):
        if cols[i].button(p, use_container_width=True, key=f"prompt_{i}"):
            st.session_state["prefill"] = p
            st.rerun()

# ------------------------------------------------------------
# SHOW CHAT HISTORY
# ------------------------------------------------------------
if mode == "🌍 LLM Chat":
    active_messages = st.session_state.messages_llm
else:
    current_doc_hash = st.session_state.doc_hash
    active_messages = st.session_state.messages_pdf_dict.get(current_doc_hash, []) if current_doc_hash else []

for msg in active_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ------------------------------------------------------------
# CHAT INPUT WITH PDF UPLOAD
# ------------------------------------------------------------
# Only show status badges in PDF RAG mode (cache and PDF are not needed in LLM Chat mode)
if mode == "📄 PDF RAG":
    cache_ready = st.session_state.index is not None
    pdf_loaded = st.session_state.pdf_path is not None and os.path.exists(st.session_state.pdf_path)
    
    # Large, prominent PDF uploader at the top
    uploaded_pdf = st.file_uploader(
        "📄 Upload PDF (max 10MB)", 
        type=["pdf"],
        help="Upload a PDF file (maximum size: 10MB). The system will process and index it for you.",
        label_visibility="visible",
        key="pdf_uploader"
    )
else:
    # LLM Chat mode - no PDF uploader, no status badges (not needed)
    uploaded_pdf = None

# Always render chat input so it stays visible
# Dynamic placeholder based on mode
placeholder = "Ask any question..." if mode == "🌍 LLM Chat" else "Ask something from the PDF..."
question = st.chat_input(placeholder, key="chat")

# PDF processing (check if uploaded_pdf exists from the column above)
if uploaded_pdf:
    # Check file size (10MB limit)
    file_size_mb = len(uploaded_pdf.read()) / (1024 * 1024)
    uploaded_pdf.seek(0)  # Reset file pointer
    
    if file_size_mb > 10:
        st.error(f"❌ File too large! Size: {file_size_mb:.2f}MB. Maximum allowed: 10MB")
    else:
        try:
            # save file locally
            os.makedirs("uploads", exist_ok=True)
            # Use a safe filename (remove special characters)
            safe_filename = "".join(c for c in uploaded_pdf.name if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
            saved_path = os.path.join("uploads", safe_filename)

            with open(saved_path, "wb") as f:
                f.write(uploaded_pdf.read())

            st.session_state.pdf_path = saved_path

            with st.spinner("📌 Processing PDF and building index (this may take a minute)..."):
                try:
                    doc_hash, index, chunks, embedder = build_or_load_index(
                        saved_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )

                    st.session_state.doc_hash = doc_hash
                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.embedder = embedder

                    st.success(f"✅ PDF Ready! ({uploaded_pdf.name}) - {len(chunks)} chunks indexed")
                except Exception as cache_error:
                    # If cache loading fails, try rebuilding
                    st.warning(f"⚠️ Cache issue detected. Rebuilding cache...")
                    try:
                        # Force rebuild by deleting cache files
                        import shutil
                        doc_hash = step6_get_doc_hash_from_path(saved_path)
                        idx_path, meta_path = step6_cache_paths(doc_hash)
                        if os.path.exists(idx_path):
                            os.remove(idx_path)
                        if os.path.exists(meta_path):
                            os.remove(meta_path)
                        
                        # Rebuild
                        doc_hash, index, chunks, embedder = build_or_load_index(
                            saved_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                        )
                        
                        st.session_state.doc_hash = doc_hash
                        st.session_state.index = index
                        st.session_state.chunks = chunks
                        st.session_state.embedder = embedder
                        
                        st.success(f"✅ PDF Ready! ({uploaded_pdf.name}) - {len(chunks)} chunks indexed (cache rebuilt)")
                    except Exception as rebuild_error:
                        st.error(f"❌ Error processing PDF: {str(rebuild_error)}")
                        import traceback
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    st.info("💡 Tip: Make sure the PDF is not corrupted and is readable.")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"❌ Error saving PDF: {str(e)}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

# Handle prefill: if a prompt button was clicked, use it as the question
prefill = st.session_state.get("prefill", "")
if prefill and not question:
    # Use prefill as question and clear it
    question = prefill
    st.session_state["prefill"] = ""

if question:
    # add user msg
    if mode == "🌍 LLM Chat":
        st.session_state.messages_llm.append({"role": "user", "content": question})
    else:
        current_doc_hash = st.session_state.doc_hash
        if current_doc_hash:
            st.session_state.messages_pdf_dict.setdefault(current_doc_hash, []).append(
                {"role": "user", "content": question}
            )
    with st.chat_message("user"):
        st.markdown(question)

    # assistant response
    if mode == "📄 PDF RAG" and st.session_state.index and st.session_state.embedder:
        status_placeholder = st.empty()
        status_placeholder.caption("🔎 Retrieving relevant chunks...")

    with st.chat_message("assistant"):
        if mode == "🌍 LLM Chat":
            # Show status text without dimming the page
            status_placeholder = st.empty()
            status_placeholder.info("💭 Thinking...")
            
            # Get conversation history (last 10 messages for context)
            conversation_history = st.session_state.messages_llm[-10:] if len(st.session_state.messages_llm) > 0 else []
            ans = step5_general_knowledge_answer(question, conversation_history=conversation_history, max_tokens=max_tokens)
            
            status_placeholder.empty()
            # Stream the answer with typing effect
            answer_placeholder = st.empty()
            for chunk in stream_text(ans, speed=0.02):
                answer_placeholder.markdown(chunk)
            st.session_state.messages_llm.append({"role": "assistant", "content": ans})

        else:
            # PDF RAG mode
            if not st.session_state.index or not st.session_state.embedder:
                st.warning("Upload a PDF first.")
            else:
                # Create a rebuild callback for Streamlit (captures chunk_size and chunk_overlap from sidebar)
                def rebuild_callback():
                    # Rebuild cache if normalization issue detected
                    pages = step1_read_pdf_custom(st.session_state.pdf_path)
                    chunks = step2_split_into_chunks(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    embedder, embeddings = step3_create_embeddings(chunks)
                    index = step4_build_faiss_index(embeddings)
                    doc_hash = step6_get_doc_hash_from_path(st.session_state.pdf_path)
                    idx_path, meta_path = step6_cache_paths(doc_hash)
                    step6_save_cache(index, chunks, idx_path, meta_path)
                    st.session_state.index = index
                    st.session_state.chunks = chunks
                    st.session_state.embedder = embedder
                    st.warning("Cache rebuilt due to normalization issue. Please try your question again.")
                    st.rerun()
                
                # Pass None for callback if we don't want auto-rebuild in Streamlit
                # (or pass rebuild_callback if we want automatic rebuild)
                retrieved, best_similarity = step4_retrieve_top_chunks(
                    question, st.session_state.embedder, st.session_state.index,
                    st.session_state.chunks, top_k=top_k, force_rebuild_callback=None
                )

                # Get conversation history (last 10 messages for context)
                current_doc_hash = st.session_state.doc_hash
                current_pdf_msgs = st.session_state.messages_pdf_dict.get(current_doc_hash, []) if current_doc_hash else []
                conversation_history = current_pdf_msgs[-10:] if len(current_pdf_msgs) > 0 else []
                ans = step5_answer_from_retrieved(question, retrieved, top_k=top_k, conversation_history=conversation_history, max_tokens=max_tokens)
                ans = format_pdf_answer(ans)
                status_placeholder.empty()
                # Stream the answer with typing effect
                answer_placeholder = st.empty()
                for chunk in stream_text(ans, speed=0.02):
                    answer_placeholder.markdown(chunk)
                if current_doc_hash:
                    st.session_state.messages_pdf_dict.setdefault(current_doc_hash, []).append(
                        {"role": "assistant", "content": ans}
                    )

                # sources panel
                with st.expander("📌 Sources (chunks)", expanded=False):
                    st.write(f"Best similarity: {best_similarity:.4f} | threshold: {threshold:.2f}")
                    for i, (chunk, score) in enumerate(retrieved[:top_k], start=1):
                        st.markdown(f"**#{i} — Page {chunk['page']} — sim: {score:.4f}**")
                        st.code(chunk["text"][:900])

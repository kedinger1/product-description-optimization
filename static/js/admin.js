/**
 * Feed Optimizer Admin - Common JavaScript
 */

// Format file sizes
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// Format dates
function formatDate(isoString) {
    const date = new Date(isoString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Copy to clipboard
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (err) {
        console.error('Failed to copy:', err);
        return false;
    }
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// API helper
async function api(endpoint, method = 'GET', data = null) {
    const options = {
        method,
        headers: {}
    };

    if (data) {
        options.headers['Content-Type'] = 'application/json';
        options.body = JSON.stringify(data);
    }

    const response = await fetch(endpoint, options);
    return response.json();
}

// Initialize tooltips and other UI elements
document.addEventListener('DOMContentLoaded', () => {
    // Add loading state to buttons on click
    document.querySelectorAll('.btn--primary').forEach(btn => {
        btn.addEventListener('click', function() {
            if (this.dataset.noLoading) return;
            const originalText = this.innerHTML;
            // Only show loading for async operations
        });
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + S to save
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            const saveBtn = document.querySelector('[onclick*="save"]');
            if (saveBtn) saveBtn.click();
        }
    });
});

// Console welcome message
console.log('%cFeed Optimizer Admin', 'color: #e94560; font-size: 20px; font-weight: bold;');
console.log('%cThe 1916 Company', 'color: #94a3b8; font-size: 12px;');

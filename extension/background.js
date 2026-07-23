/**
 * background.js — Service Worker
 * Forwards analysis requests from the popup to the FastAPI backend
 * running at http://127.0.0.1:8000.
 */

const API_BASE = "http://127.0.0.1:8000";

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === "analyze_article") {

        fetch(`${API_BASE}/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: request.data.text,
                url: request.data.url || null,
            }),
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(result => {
                sendResponse({ success: true, result: result });
            })
            .catch(error => {
                console.error("Backend API Error:", error);
                sendResponse({ success: false, error: error.message });
            });

        return true;  // Keep message channel open for async response
    }

    if (request.type === "check_health") {
        fetch(`${API_BASE}/health`)
            .then(r => r.json())
            .then(data => sendResponse({ success: true, data }))
            .catch(err => sendResponse({ success: false, error: err.message }));
        return true;
    }
});

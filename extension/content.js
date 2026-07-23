/**
 * content.js — Article Text Extraction (Content Script)
 * Extracts article text from the current page using cascading CSS selectors
 * for robust extraction across different news site layouts.
 */

function extractArticleData() {
    // Extract title 
    let title = "";

    // Priority: <h1> in article context → standalone <h1> → <title>
    const articleH1 = document.querySelector("article h1, .article h1, .post h1");
    const anyH1 = document.querySelector("h1");

    if (articleH1) {
        title = articleH1.innerText.trim();
    } else if (anyH1) {
        title = anyH1.innerText.trim();
    } else {
        title = document.title || "";
    }

    // Extract body text 
    // Cascading selectors: article-specific → main → generic
    const selectorCascade = [
        "article p",
        "[role='article'] p",
        "main article p",
        ".article-body p",
        ".article-content p",
        ".story-body p",
        ".post-content p",
        "main p",
        ".content p",
        "#content p",
    ];

    let paragraphs = [];

    for (const selector of selectorCascade) {
        const found = document.querySelectorAll(selector);
        if (found.length >= 3) {  // Need at least 3 paragraphs to consider it an article
            paragraphs = found;
            break;
        }
    }

    // Final fallback: all <p> tags
    if (paragraphs.length === 0) {
        paragraphs = document.querySelectorAll("p");
    }

    let content = "";
    paragraphs.forEach(p => {
        const text = p.innerText.trim();
        // Filter out navigation items, very short fragments, and common noise
        if (text.length > 30 && !isBoilerplate(text)) {
            content += text + "\n";
        }
    });

    // Extract metadata
    let author = "";
    const authorMeta = document.querySelector(
        'meta[name="author"], meta[property="article:author"]'
    );
    if (authorMeta) {
        author = authorMeta.getAttribute("content") || "";
    }

    return {
        url: window.location.href,
        title: title,
        text: content.substring(0, 15000),  // Cap at 15K chars
        author: author,
    };
}

/**
 Checks if a text fragment is likely boilerplate (nav, footer, etc.)
 */
function isBoilerplate(text) {
    const lower = text.toLowerCase();
    const boilerplatePatterns = [
        "subscribe to", "sign up for", "newsletter",
        "cookie", "privacy policy", "terms of service",
        "all rights reserved", "copyright ©",
        "share on facebook", "share on twitter",
        "advertisement", "sponsored",
        "click here", "read more",
        "follow us on",
    ];
    return boilerplatePatterns.some(pattern => lower.includes(pattern));
}

// Message listener
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === "extract_article") {
        const data = extractArticleData();
        sendResponse(data);
    }
    return true;  // Async response
});

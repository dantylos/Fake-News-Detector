/**
 * popup.js — Extension Popup Logic
 * Orchestrates: content extraction → backend analysis → result display.
 * Handles the full response schema including SHAP sentence explanations.
 */

// Confidence tier constants 
// Thresholds applied to the confidence of the PREDICTED class.
// Adjust these after empirical calibration on the validation set.
const CONFIDENCE_TIERS = {
    HIGH_MIN: 0.85,   // 85-99 %
    MID_MIN: 0.65,   // 65-84 %
    // Below MID_MIN (40-64 %) → inconclusive gray zone

    FAKE_HIGH: { label: "High misinformation chance", color: "#E24B4A" },
    FAKE_MID: { label: "Likely unreliable", color: "#EF9F27" },
    REAL_HIGH: { label: "Likely credible", color: "#639922" },
    REAL_MID: { label: "Probably credible", color: "#1D9E75" },
    INCONCLUSIVE: {
        label: "Inconclusive \u2014 model not confident",
        color: "#888780",
    },
};

/**
 * Returns { label, color, verdict, verdictClass, detail } based on
 * the predicted class and its confidence percentage (0-1 scale).
 */
function getConfidenceTier(isFake, displayConfidence) {
    const pct = displayConfidence * 100;

    if (pct >= CONFIDENCE_TIERS.HIGH_MIN * 100) {
        // 85-99 %
        if (isFake) {
            return {
                ...CONFIDENCE_TIERS.FAKE_HIGH,
                verdict: "Likely Fake",
                verdictClass: "fake",
                detail: "This article exhibits patterns commonly associated with misleading content.",
            };
        }
        return {
            ...CONFIDENCE_TIERS.REAL_HIGH,
            verdict: "Likely Real",
            verdictClass: "real",
            detail: "This article appears consistent with credible reporting patterns.",
        };
    }

    if (pct >= CONFIDENCE_TIERS.MID_MIN * 100) {
        // 65-84 %
        if (isFake) {
            return {
                ...CONFIDENCE_TIERS.FAKE_MID,
                verdict: "Likely Fake",
                verdictClass: "fake",
                detail: "Some patterns suggest this article may be unreliable.",
            };
        }
        return {
            ...CONFIDENCE_TIERS.REAL_MID,
            verdict: "Probably Real",
            verdictClass: "real",
            detail: "This article is probably credible, but verify with other sources.",
        };
    }

    // 40-64 % — inconclusive gray zone (no fake/real verdict)
    return {
        ...CONFIDENCE_TIERS.INCONCLUSIVE,
        verdict: "Inconclusive",
        verdictClass: "inconclusive",
        detail: "The model is not confident enough to make a determination. Cross-reference with trusted sources.",
    };
}

document.addEventListener("DOMContentLoaded", () => {
    // DOM references 
    const analyzeBtn = document.getElementById("analyze-btn");
    const resetBtn = document.getElementById("reset-btn");
    const retryBtn = document.getElementById("retry-btn");

    const initialState = document.getElementById("initial-state");
    const loadingState = document.getElementById("loading-state");
    const resultState = document.getElementById("result-state");
    const errorState = document.getElementById("error-state");

    const apiStatus = document.getElementById("api-status");
    const statusText = apiStatus.querySelector(".status-text");

    const scoreValue = document.getElementById("score-value");
    const scoreLabel = document.getElementById("score-label");
    const ringProgress = document.getElementById("ring-progress");
    const verdict = document.getElementById("verdict");
    const verdictDetail = document.getElementById("verdict-detail");
    const machineEvalSection = document.getElementById("machine-evaluation-section");
    const machineEvalText = document.getElementById("machine-evaluation-text");
    const sentenceList = document.getElementById("sentence-list");
    const modelInfo = document.getElementById("model-info");
    const disclaimer = document.getElementById("disclaimer").querySelector("span");
    const errorText = document.getElementById("error-text");

    // State management 
    function showState(state) {
        [initialState, loadingState, resultState, errorState].forEach(s =>
            s.classList.add("hidden")
        );
        state.classList.remove("hidden");
    }

    // API health check 
    function checkApiHealth() {
        chrome.runtime.sendMessage({ type: "check_health" }, (response) => {
            if (response && response.success) {
                apiStatus.className = "status-badge online";
                statusText.textContent = "API Online";
            } else {
                apiStatus.className = "status-badge offline";
                statusText.textContent = "API Offline";
            }
        });
    }

    // Check health on popup open
    checkApiHealth();

    // Analyze flow 
    function analyzeCurrentPage() {
        showState(loadingState);

        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (!tabs[0]) {
                showError("Cannot access the current tab.");
                return;
            }

            // Send message to content script
            chrome.tabs.sendMessage(
                tabs[0].id,
                { type: "extract_article" },
                (response) => {
                    if (chrome.runtime.lastError) {
                        showError(
                            "Cannot analyze this page. Make sure you're on " +
                            "a news article (not a browser internal page)."
                        );
                        return;
                    }

                    if (response && response.text && response.text.trim().length > 50) {
                        sendToBackend(response);
                    } else {
                        showError(
                            "Could not extract enough article text from this page. " +
                            "Try a page with a full news article."
                        );
                    }
                }
            );
        });
    }

    function sendToBackend(articleData) {
        chrome.runtime.sendMessage(
            { type: "analyze_article", data: articleData },
            (response) => {
                if (response && response.success) {
                    displayResult(response.result);
                } else {
                    const detail = response?.error || "Unknown error";
                    showError(
                        `Backend analysis failed: ${detail}\n\n` +
                        "Is the API running? Start it with:\n" +
                        "uvicorn src.api.main:app --port 8000"
                    );
                }
            }
        );
    }

    // Display result 
    function displayResult(result) {
        showState(resultState);

        const displayConfidence = result.confidence || 0;
        const isFake = result.prediction === "fake";
        const pct = Math.round(displayConfidence * 100);

        // Look up the tier for this prediction + confidence
        const tier = getConfidenceTier(isFake, displayConfidence);

        // Animate confidence ring 
        const circumference = 2 * Math.PI * 52; // r=52 from SVG
        const offset = circumference * (1 - displayConfidence);

        ringProgress.style.stroke = tier.color;

        setTimeout(() => {
            ringProgress.style.strokeDashoffset = offset;
        }, 100);

        // Score text
        animateCounter(scoreValue, pct);
        scoreLabel.textContent = tier.label;

        // Verdict
        verdict.textContent = tier.verdict;
        verdict.className = `verdict ${tier.verdictClass}`;
        verdictDetail.textContent = tier.detail;

        // Machine Evaluation 
        if (result.machine_evaluation) {
            machineEvalSection.classList.remove("hidden");
            machineEvalText.textContent = result.machine_evaluation;
        } else {
            machineEvalSection.classList.add("hidden");
        }

        // SHAP sentences 
        sentenceList.innerHTML = "";
        const topSentences = result.explanation?.top_sentences || [];

        if (topSentences.length > 0) {
            document.getElementById("explanation-section").classList.remove("hidden");

            // Find max |shap_value| for normalization
            const maxAbs = Math.max(
                ...topSentences.map(s => Math.abs(s.shap_value)),
                0.01
            );

            topSentences.forEach((item, i) => {
                const li = document.createElement("li");
                li.className = "sentence-item";
                li.style.animationDelay = `${i * 0.1}s`;

                const isPositive = item.shap_value > 0;
                const barWidth = Math.min(
                    Math.round((Math.abs(item.shap_value) / maxAbs) * 100),
                    100
                );

                li.innerHTML = `
                    <div class="shap-bar-wrapper">
                        <div class="shap-bar">
                            <div class="shap-bar-fill ${isPositive ? 'positive' : 'negative'}"
                                 style="width: ${barWidth}%"></div>
                        </div>
                        <span class="shap-value ${isPositive ? 'positive' : 'negative'}">
                            ${item.shap_value > 0 ? '+' : ''}${item.shap_value.toFixed(3)}
                        </span>
                    </div>
                    <span class="sentence-text">${truncateText(item.sentence, 120)}</span>
                `;
                sentenceList.appendChild(li);
            });
        } else {
            document.getElementById("explanation-section").classList.add("hidden");
        }

        // Model info 
        modelInfo.textContent = `Model: ${result.model_version || "unknown"} • ${result.timestamp || ""}`;

        // Disclaimer 
        disclaimer.textContent = result.disclaimer ||
            "This is an automated assessment and not a definitive truth judgment.";
    }

    // Helpers 
    function animateCounter(element, target) {
        let current = 0;
        const step = Math.max(1, Math.ceil(target / 30));
        const interval = setInterval(() => {
            current += step;
            if (current >= target) {
                current = target;
                clearInterval(interval);
            }
            element.textContent = `${current}%`;
        }, 30);
    }

    function truncateText(text, maxLen) {
        if (text.length <= maxLen) return text;
        return text.substring(0, maxLen) + "\u2026";
    }

    function showError(msg) {
        showState(errorState);
        errorText.textContent = msg;
    }

    // Event listeners 
    analyzeBtn.addEventListener("click", analyzeCurrentPage);
    resetBtn.addEventListener("click", () => {
        showState(initialState);
        // Reset ring animation
        ringProgress.style.strokeDashoffset = 326.73;
        checkApiHealth();
    });
    retryBtn.addEventListener("click", () => {
        showState(initialState);
        checkApiHealth();
    });
});

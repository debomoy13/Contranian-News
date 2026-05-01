document.addEventListener('DOMContentLoaded', () => {
    const tabBrowse = document.getElementById('tab-browse');
    const tabPaste = document.getElementById('tab-paste');
    const sectionBrowse = document.getElementById('section-browse');
    const sectionPaste = document.getElementById('section-paste');

    tabBrowse.addEventListener('click', () => {
        tabBrowse.classList.add('active');
        tabPaste.classList.remove('active');
        sectionBrowse.style.display = 'block';
        sectionPaste.style.display = 'none';
    });

    tabPaste.addEventListener('click', () => {
        tabPaste.classList.add('active');
        tabBrowse.classList.remove('active');
        sectionPaste.style.display = 'block';
        sectionBrowse.style.display = 'none';
    });

    // API Integration for Paste Text
    const analyzeBtn = document.getElementById('analyze-btn');
    const textInput = document.getElementById('custom-text-input');
    const resultsContainer = document.getElementById('results-container');

    const getSentimentLabel = (score) => {
        if (score === 1) return { label: 'Positive', class: 'sentiment-positive' };
        if (score === -1) return { label: 'Negative', class: 'sentiment-negative' };
        return { label: 'Neutral', class: 'sentiment-neutral' };
    };

    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) {
            alert("Please paste some text to analyze.");
            return;
        }

        analyzeBtn.textContent = 'Analyzing...';
        analyzeBtn.disabled = true;
        resultsContainer.style.display = 'none';
        resultsContainer.innerHTML = '';

        try {
            const response = await fetch('http://127.0.0.1:5000/api/analyze-text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            // Build Results UI
            const inputSentiment = getSentimentLabel(data.input_predicted);
            
            let html = `
                <div class="input-analysis">
                    <p><strong>Your Input Sentiment:</strong> <span class="sentiment-badge ${inputSentiment.class}">${inputSentiment.label}</span></p>
                </div>
                <h3 class="results-header">Contrarian Viewpoints Found</h3>
            `;

            if (data.results && data.results.length > 0) {
                let foundValid = false;
                data.results.forEach(article => {
                    if (article.similarity >= 0.6) {
                        foundValid = true;
                        const articleSentiment = getSentimentLabel(article.predicted);
                        const similarityPct = (article.similarity * 100).toFixed(1);
                        
                        html += `
                            <div class="result-card">
                                <h4 class="result-title">${article.title}</h4>
                                <div class="result-meta">
                                    <span><span class="sentiment-badge ${articleSentiment.class}">${articleSentiment.label}</span></span>
                                    <span>Relevance: ${similarityPct}%</span>
                                </div>
                                <a href="${article.url}" target="_blank" class="read-more">Read Full Article →</a>
                            </div>
                        `;
                    }
                });

                if (!foundValid) {
                    html += `<p>No relevant contrarian viewpoints found for this specific topic.</p>`;
                }
            } else {
                html += `<p>No contrarian viewpoints found for this topic in the database.</p>`;
            }

            resultsContainer.innerHTML = html;
            resultsContainer.style.display = 'block';

        } catch (error) {
            console.error('Error analyzing text:', error);
            resultsContainer.innerHTML = `<p style="color: red;">Failed to analyze text. Please ensure the backend is running.</p>`;
            resultsContainer.style.display = 'block';
        } finally {
            analyzeBtn.textContent = 'Discover Opposing Views';
            analyzeBtn.disabled = false;
        }
    });
});

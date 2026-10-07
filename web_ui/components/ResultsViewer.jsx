"use client";

function ResultCard({ badge, badgeClass, title, description, imgSrc, imgStyle }) {
  const handleDownload = () => {
    const a = document.createElement('a');
    a.href = imgSrc;
    a.download = `${title.toLowerCase().replace(/\s+/g,'-')}.png`;
    a.click();
  };

  const handleFullSize = () => {
    fetch(imgSrc)
      .then(res => res.blob())
      .then(blob => {
        const url = URL.createObjectURL(blob);
        window.open(url, '_blank');
      });
  };

  return (
    <div className="result-card">
      <div className="result-card-img-wrap">
        <img src={imgSrc} alt={title} style={imgStyle} />
        <span className={`result-card-badge ${badgeClass}`}>{badge}</span>
      </div>
      <div className="result-card-body">
        <p className="result-card-title">{title}</p>
        <p className="result-card-desc">{description}</p>
      </div>
      <div className="result-card-actions">
        <button className="result-action-btn" onClick={handleDownload}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          Download
        </button>
        <button className="result-action-btn" onClick={handleFullSize}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
            <polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>
          </svg>
          Full Size
        </button>
      </div>
    </div>
  );
}

export default function ResultsViewer({ results }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

      {/* ─ Before / After Comparison ─ */}
      {(results.restored || results.colorized) && (
        <div className="comparison-wrapper">
          <div className="comparison-header">
            <span className="comparison-title">◈ Before & After Comparison</span>
            <span style={{ fontSize: '0.7rem', color: 'var(--gold-dim)', letterSpacing: '1px', fontFamily: 'var(--font-display)' }}>
              {results.restored ? 'RESTORATION ANALYSIS' : 'COLORIZATION ANALYSIS'}
            </span>
          </div>
          <div className="comparison-body">
            <div className="comparison-panel">
              <span className="comparison-label left">Original</span>
              <img src={results.original} alt="Original damaged photograph" />
            </div>
            <div className="comparison-panel">
              <span className="comparison-label right">
                {results.restored ? 'Restored' : 'Colorized'}
              </span>
              <img src={results.colorized || results.restored} alt="Enhanced photograph" />
            </div>
            <div className="comparison-divider" />
          </div>
        </div>
      )}

      {/* ─ Results Grid ─ */}
      <div>
        <div className="ornament" style={{ marginBottom: '1.25rem' }}>
          <span className="ornament-text">❧</span>
          <span className="results-section-title">All Outputs</span>
          <span className="ornament-text">❧</span>
        </div>

        <div className="results-grid">
          {results.restored && (
            <ResultCard
              badge="Restored"
              badgeClass="badge-restored"
              title="Structural Restoration"
              description="Overall quality improvement via global photo restoration network."
              imgSrc={results.restored}
            />
          )}

          {results.colorized && (
            <ResultCard
              badge="Colorized"
              badgeClass="badge-colorized"
              title="AI Colorization"
              description="Siggraph17 deep learning colorization applied to the restored image."
              imgSrc={results.colorized}
            />
          )}

          {results.uncertainty && (
            <ResultCard
              badge="Uncertainty"
              badgeClass="badge-uncertainty"
              title="Raw Uncertainty Map"
              description="Predicted reconstruction error per region. Dark/black = AI is confident; orange/yellow = AI was guessing here (INFERNO scale)."
              imgSrc={results.uncertainty}
            />
          )}

          {results.confidence && (
            <ResultCard
              badge="Confidence"
              badgeClass="badge-confidence"
              title="Calibrated Confidence"
              description="Per-region model confidence. Blue = high confidence (trusted reconstruction); red = low confidence (AI-fabricated detail)."
              imgSrc={results.confidence}
            />
          )}
        </div>

        {/* ─ Uncertainty & Confidence Insight ─ */}
        {(results.uncertainty_reason || results.confidence_reason) && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '1rem',
            marginTop: '1rem'
          }}>
            {results.uncertainty_reason && (
              <div style={{
                background: 'rgba(255,200,80,0.06)',
                border: '1px solid rgba(255,200,80,0.2)',
                borderRadius: '8px',
                padding: '0.9rem 1.1rem',
                display: 'flex',
                gap: '0.65rem',
                alignItems: 'flex-start',
              }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#c8a84b" strokeWidth="2"
                  style={{ flexShrink: 0, marginTop: '2px' }}>
                  <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
                  <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                <div>
                  <p style={{ margin: '0 0 0.2rem', fontSize: '0.7rem', letterSpacing: '1.5px', color: '#c8a84b', fontFamily: 'var(--font-display)', textTransform: 'uppercase' }}>
                    Why this uncertainty?
                  </p>
                  <p style={{ margin: 0, fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: '1.55' }}>
                    {results.uncertainty_reason}
                  </p>
                </div>
              </div>
            )}

            {results.confidence_reason && (
              <div style={{
                background: 'rgba(80,180,255,0.06)',
                border: '1px solid rgba(80,180,255,0.2)',
                borderRadius: '8px',
                padding: '0.9rem 1.1rem',
                display: 'flex',
                gap: '0.65rem',
                alignItems: 'flex-start',
              }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#50b4ff" strokeWidth="2"
                  style={{ flexShrink: 0, marginTop: '2px' }}>
                  <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
                  <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                <div>
                  <p style={{ margin: '0 0 0.2rem', fontSize: '0.7rem', letterSpacing: '1.5px', color: '#50b4ff', fontFamily: 'var(--font-display)', textTransform: 'uppercase' }}>
                    Why this confidence?
                  </p>
                  <p style={{ margin: 0, fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: '1.55' }}>
                    {results.confidence_reason}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ─ Disclaimer ─ */}
      <div className="disclaimer">
        <p>
          <strong>Scholarly Disclaimer:</strong> Regions flagged in the uncertainty map contain AI-generated reconstructions
          that are visually plausible but cannot be independently verified against historical fact.
          These outputs must <strong>not</strong> be presented as authentic archival records without corroborating primary sources.
          Consult the confidence map to evaluate which regions of the image are reliably reconstructed.
        </p>
      </div>
    </div>
  );
}

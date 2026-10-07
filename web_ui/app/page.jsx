"use client";

import { useState, useCallback } from 'react';
import ImageUploader from '../components/ImageUploader';
import OptionToggle from '../components/OptionToggle';
import ProgressPanel from '../components/ProgressPanel';
import ResultsViewer from '../components/ResultsViewer';

const FILM_HOLES = Array.from({ length: 40 });

// ─── Simulate the 4-stage pipeline with realistic delays ──────────
async function simulatePipeline(options, onStageChange, onDone) {
  const stages = ['stage1'];
  if (options.colorization) stages.push('colorize');
  stages.push('faces', 'blend');
  if (options.uncertainty) stages.push('uncertainty');

  const done = [];

  for (const stage of stages) {
    onStageChange(stage, done);
    await new Promise(r => setTimeout(r, 1500 + Math.random() * 1000));
    done.push(stage);
  }
  onDone();
}

// ─── NavBar ────────────────────────────────────────────────────────
function NavBar() {
  return (
    <nav className="topnav">
      <a href="/" className="nav-logo" style={{ textDecoration: 'none' }}>
        {/* Hourglass icon */}
        <svg className="nav-logo-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2">
          <path d="M5 22h14M5 2h14M17 22v-4.172a2 2 0 0 0-.586-1.414L12 12l-4.414 4.414A2 2 0 0 0 7 17.828V22M7 2v4.172a2 2 0 0 0 .586 1.414L12 12l4.414-4.414A2 2 0 0 0 17 6.172V2"/>
        </svg>
        <div className="nav-logo-text">
          Timekeeper
          <span>Historical Photo Restoration</span>
        </div>
      </a>

      <ul className="nav-links">
        <li><a href="#">Restore</a></li>
        <li><a href="#">About</a></li>
        <li><a href="#">Research</a></li>
        <li><a href="https://github.com/Nihir-Soni/TIMEKEEPER" target="_blank" rel="noopener">GitHub</a></li>
      </ul>
    </nav>
  );
}

// ─── Sidebar ────────────────────────────────────────────────────────
function Sidebar({ image, setImage, options, onOptionChange, saturation, setSaturation, onRestore, isProcessing, statusMsg, statusType }) {
  return (
    <aside className="sidebar">

      {/* Upload */}
      <div className="sidebar-section">
        <p className="sidebar-section-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
          Photograph Input
        </p>
        <div className="sidebar-section-body">
          <ImageUploader onImageSelected={setImage} currentImage={image} />
        </div>
      </div>

      {/* Restoration Options */}
      <div className="sidebar-section">
        <p className="sidebar-section-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/></svg>
          Restoration Options
        </p>
        <div className="sidebar-section-body" style={{padding:'0.5rem 1rem'}}>
          <OptionToggle
            name="scratchRemoval"
            label="Scratch & Stain Removal"
            description="Detect and remove physical damage from the photo"
            checked={options.scratchRemoval}
            onChange={onOptionChange}
          />
          <OptionToggle
            name="colorization"
            label="AI Colorization"
            description="Siggraph17 deep-learned colorization"
            checked={options.colorization}
            onChange={onOptionChange}
          />
          <OptionToggle
            name="highResolution"
            label="High Resolution Mode"
            description="Enhanced face detection at full scale"
            checked={options.highResolution}
            onChange={onOptionChange}
          />
          <OptionToggle
            name="uncertainty"
            label="Uncertainty Estimation"
            description="Regional confidence maps via calibrated U-Net"
            checked={options.uncertainty}
            onChange={onOptionChange}
          />
        </div>
      </div>

      {/* Saturation */}
      {options.colorization && (
        <div className="sidebar-section">
          <p className="sidebar-section-title">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
            Color Saturation
          </p>
          <div className="sidebar-section-body">
            <div className="slider-row">
              <div className="slider-label">
                <span>Intensity</span>
                <strong>{saturation.toFixed(1)}×</strong>
              </div>
              <input
                type="range"
                min="0" max="2" step="0.1"
                value={saturation}
                onChange={(e) => setSaturation(parseFloat(e.target.value))}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.25rem' }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--faded)' }}>Desaturated</span>
                <span style={{ fontSize: '0.68rem', color: 'var(--faded)' }}>Vivid</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Action */}
      <div>
        <button
          className={`restore-btn ${isProcessing ? 'loading' : ''}`}
          onClick={onRestore}
          disabled={!image || isProcessing}
          id="restore-photo-btn"
        >
          {isProcessing ? '⟳  Restoring…' : '✦  Restore Photograph'}
        </button>

        <div className="status-bar mt-1">
          <div className={`status-dot ${isProcessing ? 'active' : statusMsg.startsWith('Error') ? 'error' : ''}`} />
          <span>{statusMsg || 'Awaiting photograph…'}</span>
        </div>
      </div>

      {/* Project Info */}
      <div className="sidebar-section">
        <p className="sidebar-section-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
          About This Project
        </p>
        <div className="sidebar-section-body">
          <p style={{ fontSize: '0.78rem', color: 'var(--gold-dim)', lineHeight: 1.6, fontFamily: 'var(--font-fell)', fontStyle: 'italic' }}>
            A capstone research project combining deep photo restoration with
            calibrated regional uncertainty estimation to ensure scholarly
            integrity of AI-reconstructed historical imagery.
          </p>
          <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {['PyTorch', 'Siggraph17', 'DLIB', 'SPADE', 'U-Net'].map(t => (
              <span key={t} style={{
                fontSize: '0.65rem', padding: '2px 8px', border: '1px solid var(--gold-dim)',
                borderRadius: '2px', color: 'var(--gold-dim)', fontFamily: 'var(--font-ui)',
                letterSpacing: '0.5px'
              }}>{t}</span>
            ))}
          </div>
        </div>
      </div>
    </aside>
  );
}

// ─── Empty State ────────────────────────────────────────────────────
function EmptyState() {
  return (
    <div className="empty-state">
      <svg className="empty-state-icon" width="96" height="96" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="0.75">
        <rect x="2" y="3" width="20" height="18" rx="2"/>
        <circle cx="8.5" cy="8.5" r="2"/>
        <path d="M20 14.5l-4-4-6 7"/>
      </svg>
      <h2 className="empty-state-title">Awaiting Your Photograph</h2>
      <p className="empty-state-sub">
        Upload a damaged, aged, or faded historical photograph from the panel on the left,
        select your restoration preferences, and press <em>Restore Photograph</em>.
      </p>
      <p style={{ fontSize: '0.8rem', color: 'var(--faded)', fontFamily: 'var(--font-display)', fontStyle: 'italic', marginTop: '0.5rem' }}>
        "In every photograph there is a story waiting to be retold."
      </p>
    </div>
  );
}

// ─── Main Page ─────────────────────────────────────────────────────
export default function Home() {
  const [image, setImage] = useState(null);
  const [options, setOptions] = useState({
    scratchRemoval: false,
    colorization: true,
    highResolution: false,
    uncertainty: true,
  });
  const [saturation, setSaturation] = useState(1.0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeStage, setActiveStage] = useState(null);
  const [doneStages, setDoneStages] = useState([]);
  const [results, setResults] = useState(null);
  const [statusMsg, setStatusMsg] = useState('');

  const handleOptionChange = (e) => {
    const { name, checked } = e.target;
    setOptions(prev => ({ ...prev, [name]: checked }));
  };

  const handleRestore = useCallback(async () => {
    if (!image || isProcessing) return;

    setIsProcessing(true);
    setResults(null);
    setDoneStages([]);
    setActiveStage('stage1');
    setStatusMsg('Stage 1/4: Restoring overall quality…');

    const stageMessages = {
      stage1:      'Stage 1/4: Restoring overall quality…',
      colorize:    'Colorizing… (Siggraph17)',
      faces:       'Stage 2-3/4: Detecting & enhancing faces…',
      blend:       'Stage 4/4: Blending and finalizing…',
      uncertainty: 'Running uncertainty analysis…',
    };

    // await simulatePipeline(
    //   options,
    //   (stage, done) => {
    //     setActiveStage(stage);
    //     setDoneStages([...done]);
    //     setStatusMsg(stageMessages[stage] || 'Processing…');
    //   },
    //   () => {
    //     setResults({
    //       original: image,
    //       restored: image,
    //       colorized: options.colorization ? image : null,
    //       uncertainty: options.uncertainty ? image : null,
    //       confidence: options.uncertainty ? image : null,
    //     });
    //     setIsProcessing(false);
    //     setActiveStage(null);
    //     setStatusMsg('✓ All processing done!');
    //   }
    // );

    try {
      // Convert base64 image back to file to send to API
      const res = await fetch(image);
      const blob = await res.blob();
      const file = new File([blob], "input_image.png", { type: "image/png" });

      const formData = new FormData();
      formData.append('file', file);
      formData.append('scratchRemoval', options.scratchRemoval);
      formData.append('colorization', options.colorization);
      formData.append('highResolution', options.highResolution);
      formData.append('uncertainty', options.uncertainty);
      formData.append('saturation', saturation);

      setStatusMsg('Uploading and processing... This may take several minutes.');
      
      const response = await fetch('http://localhost:8000/restore', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Server returned an error');
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }

      setResults(data);
      setStatusMsg('✓ All processing done!');
    } catch (err) {
      console.error(err);
      setStatusMsg(`Error: ${err.message}`);
    } finally {
      setIsProcessing(false);
      setActiveStage(null);
    }
  }, [image, isProcessing, options]);

  return (
    <div className="app-shell">
      <NavBar />

      <div className="main-layout">
        <Sidebar
          image={image}
          setImage={setImage}
          options={options}
          onOptionChange={handleOptionChange}
          saturation={saturation}
          setSaturation={setSaturation}
          onRestore={handleRestore}
          isProcessing={isProcessing}
          statusMsg={statusMsg}
        />

        <main className="content-area">
          {isProcessing ? (
            <ProgressPanel
              activeStage={activeStage}
              doneStages={doneStages}
              options={options}
            />
          ) : results ? (
            <ResultsViewer results={results} />
          ) : (
            <EmptyState />
          )}
        </main>
      </div>

      {/* Film Strip Footer */}
      <footer className="filmstrip">
        <div className="filmstrip-holes">
          {FILM_HOLES.map((_, i) => <div key={i} className="filmstrip-hole" />)}
        </div>
        <span className="filmstrip-text">TIMEKEEPER · CAPSTONE PROJECT · HISTORICAL PHOTO RESTORATION</span>
      </footer>
    </div>
  );
}

"use client";

const STAGES = [
  { key: 'stage1',     label: 'Quality Restoration',    sub: 'Global photo enhancement & denoise' },
  { key: 'colorize',   label: 'AI Colorization',         sub: 'Siggraph17 learned colorization' },
  { key: 'faces',      label: 'Face Detection & Enhancement', sub: 'DLIB landmark detection + SPADE' },
  { key: 'blend',      label: 'Blending & Finalizing',   sub: 'Warp-back and seamless compositing' },
  { key: 'uncertainty',label: 'Uncertainty Analysis',    sub: 'Regional confidence estimation' },
];

function StageIcon({ status }) {
  if (status === 'done') {
    return (
      <div className="stage-icon done">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
      </div>
    );
  }
  if (status === 'active') {
    return (
      <div className="stage-icon active">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <circle cx="12" cy="12" r="10"/>
          <polyline points="12 6 12 12 16 14"/>
        </svg>
      </div>
    );
  }
  return <div className="stage-icon pending">{STAGES.findIndex(s => s.key === status?.key) + 1}</div>;
}

export default function ProgressPanel({ activeStage, doneStages, options }) {
  const visibleStages = STAGES.filter(s => {
    if (s.key === 'colorize' && !options.colorization) return false;
    if (s.key === 'uncertainty' && !options.uncertainty) return false;
    return true;
  });

  return (
    <div className="progress-panel">
      <p className="progress-title">Restoring Your Photograph…</p>
      <p style={{ fontFamily: 'var(--font-fell)', fontStyle: 'italic', color: 'var(--faded)', fontSize: '0.9rem', textAlign: 'center' }}>
        Please wait while we breathe life back into history
      </p>

      <div className="progress-stages">
        {visibleStages.map((stage) => {
          const isDone = doneStages.includes(stage.key);
          const isActive = activeStage === stage.key;
          const className = `progress-stage ${isDone ? 'done' : isActive ? 'active' : 'pending'}`;
          const status = isDone ? 'done' : isActive ? 'active' : 'pending';

          return (
            <div key={stage.key} className={className}>
              <StageIcon status={isDone ? 'done' : isActive ? 'active' : 'pending'} />
              <div className="stage-text">
                <strong>{stage.label}</strong>
                <small>{stage.sub}</small>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

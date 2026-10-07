"use client";

export default function OptionToggle({ name, label, description, checked, onChange }) {
  return (
    <label className="option-row" style={{userSelect:'none'}}>
      <div className={`option-checkbox ${checked ? 'checked' : ''}`}>
        {checked && (
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
        )}
      </div>
      <input
        type="checkbox"
        name={name}
        checked={checked}
        onChange={onChange}
        style={{ display: 'none' }}
      />
      <div className="option-label">
        <strong>{label}</strong>
        <small>{description}</small>
      </div>
    </label>
  );
}

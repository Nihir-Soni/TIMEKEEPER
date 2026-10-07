"use client";
import { useRef, useState } from 'react';

export default function ImageUploader({ onImageSelected, currentImage }) {
  const fileInputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => onImageSelected(e.target.result);
    reader.readAsDataURL(file);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const onDragOver = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);

  if (currentImage) {
    return (
      <div className="upload-preview" onClick={() => fileInputRef.current.click()}>
        <img src={currentImage} alt="Selected photograph" />
        <div className="upload-preview-overlay">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
          Change Photograph
        </div>
        <input type="file" accept="image/*" ref={fileInputRef} style={{display:'none'}}
          onChange={(e) => handleFile(e.target.files[0])} />
      </div>
    );
  }

  return (
    <div
      className={`upload-zone ${dragging ? 'drag-over' : ''}`}
      onClick={() => fileInputRef.current.click()}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
    >
      <div className="upload-zone-icon">
        <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
          <circle cx="8.5" cy="8.5" r="1.5"/>
          <polyline points="21 15 16 10 5 21"/>
        </svg>
      </div>
      <p className="upload-zone-title">Select Photograph</p>
      <p className="upload-zone-sub">Click to browse or drag & drop</p>
      <p className="upload-zone-sub" style={{marginTop:'0.25rem', fontSize:'0.68rem'}}>JPG, PNG, BMP supported</p>
      <input type="file" accept="image/*" ref={fileInputRef} style={{display:'none'}}
        onChange={(e) => handleFile(e.target.files[0])} />
    </div>
  );
}

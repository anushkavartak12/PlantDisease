import { useEffect, useRef, useState } from 'react';
import type { Lang } from '../data/crops';
import { UI } from '../data/ui';
import { sfx } from '../lib/sound';

interface Props {
  lang: Lang;
  onCapture(canvas: HTMLCanvasElement): void;
}

export function CameraCapture({ lang, onCapture }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [live, setLive] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => () => streamRef.current?.getTracks().forEach((t) => t.stop()), []);

  async function openCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1440 } },
        audio: false,
      });
      streamRef.current = stream;
      videoRef.current!.srcObject = stream;
      await videoRef.current!.play();
      setLive(true);
    } catch {
      setError(true);
    }
  }

  function capture() {
    const v = videoRef.current!;
    const c = document.createElement('canvas');
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext('2d')!.drawImage(v, 0, 0);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setLive(false);
    onCapture(c);
  }

  function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      c.getContext('2d')!.drawImage(img, 0, 0);
      URL.revokeObjectURL(img.src);
      onCapture(c);
    };
    img.src = URL.createObjectURL(file);
  }

  return (
    <div className="camera">
      <div className={`viewfinder ${live ? 'on' : ''}`}>
        <video ref={videoRef} playsInline muted />
        {live && <div className="leaf-guide" />}
        {!live && <div className="placeholder"><span>🍃</span><small>{UI.scanTip[lang]}</small></div>}
      </div>
      {error && <p className="warn">{UI.noCamera[lang]}</p>}
      <div className="actions">
        {live ? (
          <button className="primary big" onClick={() => { sfx.pop(); capture(); }}>📸 {UI.capture[lang]}</button>
        ) : (
          <button className="primary big" onClick={openCamera}>📷 {UI.openCamera[lang]}</button>
        )}
        <label className="secondary">
          🖼️ {UI.upload[lang]}
          <input type="file" accept="image/*" onChange={onFile} hidden />
        </label>
      </div>
    </div>
  );
}

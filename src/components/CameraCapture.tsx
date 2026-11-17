import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Camera, RefreshCw, X, Upload, Loader2, Image as ImageIcon } from 'lucide-react';

interface CameraCaptureProps {
  onCapture: (dataUrl: string) => void;
  onClose?: () => void;
  embedded?: boolean;
  paused?: boolean;
  onUploadFile?: (file: File) => void;
  statusNote?: string;
}

export const CameraCapture: React.FC<CameraCaptureProps> = ({
  onCapture,
  onClose,
  embedded = false,
  paused = false,
  onUploadFile,
  statusNote,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nativeInputRef = useRef<HTMLInputElement | null>(null);
  const galleryInputRef = useRef<HTMLInputElement | null>(null);

  const mountedRef = useRef(true);
  const pausedRef = useRef(paused);
  const attemptIdRef = useRef(0);
  const selectedDeviceIdRef = useRef('');
  const facingModeRef = useRef<'environment' | 'user'>('environment');

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');
  const [isStarting, setIsStarting] = useState(!paused);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  useEffect(() => {
    selectedDeviceIdRef.current = selectedDeviceId;
  }, [selectedDeviceId]);

  useEffect(() => {
    facingModeRef.current = facingMode;
  }, [facingMode]);

  const stopTracks = (stream: MediaStream | null) => {
    if (!stream) return;
    stream.getTracks().forEach((track) => {
      try {
        track.stop();
      } catch {
        // ignore track stop failure
      }
    });
  };

  const stopStream = useCallback(() => {
    attemptIdRef.current += 1;
    if (streamRef.current) {
      stopTracks(streamRef.current);
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      if (!mountedRef.current) return;
      const videoInputs = all.filter((d) => d.kind === 'videoinput');
      setDevices(videoInputs);
    } catch {
      // ignore enumeration errors
    }
  }, []);

  const startCamera = useCallback(
    async (opts?: { deviceId?: string; facing?: 'environment' | 'user' }) => {
      stopStream();
      if (!mountedRef.current || pausedRef.current || document.visibilityState === 'hidden') {
        setIsStarting(false);
        return;
      }

      const myAttempt = ++attemptIdRef.current;
      setIsStarting(true);
      setErrorMsg(null);

      if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
        if (mountedRef.current && attemptIdRef.current === myAttempt) {
          setIsStarting(false);
          setErrorMsg(
            'Live camera streaming is unavailable in this browser context. Choose a photo, drop an image here, or open your device camera.'
          );
        }
        return;
      }

      const targetDeviceId =
        opts?.deviceId !== undefined ? opts.deviceId : selectedDeviceIdRef.current;
      const targetFacing = opts?.facing ?? facingModeRef.current;

      // Progressive constraint fallback chain so any phone, tablet, laptop, or webcam works
      const constraintCandidates: MediaStreamConstraints[] = targetDeviceId
        ? [
            {
              video: {
                deviceId: { exact: targetDeviceId },
                width: { ideal: 1920 },
                height: { ideal: 1080 },
              },
              audio: false,
            },
            {
              video: { deviceId: { exact: targetDeviceId } },
              audio: false,
            },
            { video: true, audio: false },
          ]
        : [
            {
              video: {
                facingMode: { ideal: targetFacing },
                width: { ideal: 1920 },
                height: { ideal: 1080 },
              },
              audio: false,
            },
            {
              video: { facingMode: targetFacing },
              audio: false,
            },
            { video: true, audio: false },
          ];

      let acquiredStream: MediaStream | null = null;
      let lastErr: any = null;

      for (const constraints of constraintCandidates) {
        if (!mountedRef.current || pausedRef.current || attemptIdRef.current !== myAttempt) {
          break;
        }
        try {
          acquiredStream = await navigator.mediaDevices.getUserMedia(constraints);
          break;
        } catch (err: any) {
          lastErr = err;
          if (
            err?.name === 'NotAllowedError' ||
            err?.name === 'PermissionDeniedError' ||
            err?.name === 'NotReadableError'
          ) {
            break;
          }
        }
      }

      // Async unmount / paused / superseded guard so tracks never leak in background
      if (!mountedRef.current || pausedRef.current || attemptIdRef.current !== myAttempt) {
        stopTracks(acquiredStream);
        if (mountedRef.current) {
          setIsStarting(false);
        }
        return;
      }

      if (!acquiredStream) {
        setIsStarting(false);
        const name = lastErr?.name || '';
        if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
          setErrorMsg(
            'Camera permission was denied. Allow camera access in browser settings, or pick/drop an image below.'
          );
        } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
          setErrorMsg(
            'No live camera was found on this device. Drop an image here, paste a screenshot, or browse your photos.'
          );
        } else if (name === 'NotReadableError') {
          setErrorMsg(
            'Camera is currently used by another app. Close the other app and tap Retry Camera, or choose a photo below.'
          );
        } else {
          setErrorMsg(
            lastErr?.message ||
              'Could not start live camera preview. Choose a photo, drop an image, or use device capture.'
          );
        }
        return;
      }

      streamRef.current = acquiredStream;

      const activeTrack = acquiredStream.getVideoTracks()[0];
      const activeSettings = activeTrack?.getSettings?.();
      if (activeSettings?.deviceId && !targetDeviceId) {
        setSelectedDeviceId(activeSettings.deviceId);
        selectedDeviceIdRef.current = activeSettings.deviceId;
      }

      await refreshDevices();
      if (!mountedRef.current || pausedRef.current || attemptIdRef.current !== myAttempt) {
        stopTracks(acquiredStream);
        streamRef.current = null;
        return;
      }

      if (videoRef.current) {
        videoRef.current.srcObject = acquiredStream;
        try {
          await videoRef.current.play();
        } catch {
          // autoPlay + playsInline will render frames on browsers blocking programmatic play
        }
      }

      if (mountedRef.current && attemptIdRef.current === myAttempt) {
        setIsStarting(false);
      }
    },
    [refreshDevices, stopStream]
  );

  // Lifecycle: mount/unmount + pause/resume when review sheet or lightbox opens
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      stopStream();
    };
  }, [stopStream]);

  useEffect(() => {
    if (paused) {
      stopStream();
      setIsStarting(false);
    } else if (document.visibilityState !== 'hidden') {
      startCamera();
    }
  }, [paused, startCamera, stopStream]);

  // Pause camera hardware when browser tab goes background; resume cleanly on return
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'hidden') {
        stopStream();
        setIsStarting(false);
      } else if (!pausedRef.current && mountedRef.current) {
        startCamera();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [startCamera, stopStream]);

  const handleSwitchDevice = (nextDeviceId: string) => {
    setSelectedDeviceId(nextDeviceId);
    selectedDeviceIdRef.current = nextDeviceId;
    startCamera({ deviceId: nextDeviceId });
  };

  const handleFlipCamera = () => {
    const nextFacing = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(nextFacing);
    facingModeRef.current = nextFacing;
    setSelectedDeviceId('');
    selectedDeviceIdRef.current = '';
    startCamera({ deviceId: '', facing: nextFacing });
  };

  const handleCaptureFrame = () => {
    const video = videoRef.current;
    if (
      isCapturing ||
      !video ||
      video.readyState < 2 ||
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      return;
    }

    setIsCapturing(true);
    try {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Always draw unmirrored raw frame so text is never backwards for OCR
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
      stopStream();
      onCapture(dataUrl);
    } finally {
      setIsCapturing(false);
    }
  };

  const processSelectedFile = (file: File) => {
    if (onUploadFile) {
      stopStream();
      onUploadFile(file);
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      const result = ev.target?.result as string;
      if (result) {
        stopStream();
        onCapture(result);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processSelectedFile(file);
    }
    e.target.value = '';
  };

  const liveReady = !isStarting && !errorMsg && !paused;

  return (
    <div className="space-y-3">
      {/* Main Viewfinder Surface (supports live camera + drag-and-drop + offline fallback) */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragOver(false);
          const file = e.dataTransfer.files?.[0];
          if (file && file.type.startsWith('image/')) {
            processSelectedFile(file);
          }
        }}
        className={`relative rounded-2xl overflow-hidden border transition-colors bg-zinc-950 aspect-[3/4] sm:aspect-[4/3] max-h-[54vh] min-h-[280px] w-full flex items-center justify-center ${
          isDragOver
            ? 'border-indigo-500 ring-2 ring-indigo-500/40'
            : 'border-zinc-200 dark:border-zinc-800'
        }`}
      >
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover transition-opacity duration-200 ${
            facingMode === 'user' ? 'scale-x-[-1]' : ''
          } ${liveReady ? 'opacity-100' : 'opacity-0'}`}
        />

        {/* Subtle alignment reticle guides when camera stream is active */}
        {liveReady && (
          <div className="pointer-events-none absolute inset-6 sm:inset-8">
            <div className="absolute top-0 left-0 w-5 h-5 border-t-2 border-l-2 border-white/55 rounded-tl-md" />
            <div className="absolute top-0 right-0 w-5 h-5 border-t-2 border-r-2 border-white/55 rounded-tr-md" />
            <div className="absolute bottom-0 left-0 w-5 h-5 border-b-2 border-l-2 border-white/55 rounded-bl-md" />
            <div className="absolute bottom-0 right-0 w-5 h-5 border-b-2 border-r-2 border-white/55 rounded-br-md" />
          </div>
        )}

        {/* Drag-and-drop overlay indicator */}
        {isDragOver && (
          <div className="absolute inset-0 z-20 bg-indigo-950/80 flex flex-col items-center justify-center gap-2 text-white p-4 text-center">
            <Upload className="w-7 h-7 text-indigo-300" />
            <p className="text-sm font-medium">Drop image to auto-scan offline</p>
          </div>
        )}

        {/* Starting / Paused state */}
        {isStarting && !errorMsg && !paused && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2.5 text-zinc-300 text-xs p-4 text-center">
            <Loader2 className="w-5 h-5 animate-spin text-indigo-400" />
            <span>Starting live camera...</span>
          </div>
        )}

        {paused && !errorMsg && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-zinc-400 text-xs p-4 text-center bg-zinc-950">
            <Camera className="w-5 h-5 opacity-60" />
            <span>Camera paused while reviewing scan</span>
          </div>
        )}

        {/* Camera unavailable / permission denied interactive fallback */}
        {errorMsg && (
          <div className="absolute inset-0 p-5 flex flex-col items-center justify-center text-center gap-3.5 bg-zinc-950/95">
            <div className="w-11 h-11 rounded-2xl bg-zinc-900 border border-zinc-800 text-zinc-300 flex items-center justify-center">
              <Camera className="w-5 h-5" />
            </div>
            <p className="text-xs sm:text-sm text-zinc-300 max-w-sm leading-relaxed">{errorMsg}</p>
            <div className="flex flex-wrap items-center justify-center gap-2">
              <button
                type="button"
                onClick={() => startCamera()}
                className="min-h-[44px] px-4 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-zinc-100 text-xs font-medium inline-flex items-center gap-1.5 whitespace-nowrap"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Retry Camera</span>
              </button>
              <button
                type="button"
                onClick={() => galleryInputRef.current?.click()}
                className="min-h-[44px] px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium inline-flex items-center gap-1.5 whitespace-nowrap"
              >
                <ImageIcon className="w-3.5 h-3.5" />
                <span>Choose Photo</span>
              </button>
              <button
                type="button"
                onClick={() => nativeInputRef.current?.click()}
                className="min-h-[44px] px-4 py-2 rounded-xl bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-zinc-200 text-xs font-medium inline-flex items-center gap-1.5 whitespace-nowrap"
              >
                <Upload className="w-3.5 h-3.5" />
                <span>Device Cam</span>
              </button>
            </div>
            <p className="text-[11px] text-zinc-500">
              Or drop an image file here or paste a screenshot (Ctrl+V)
            </p>
          </div>
        )}

        {/* Top Bar Overlay: Live status / Camera Device Picker / Optional Close */}
        <div className="absolute top-2.5 left-2.5 right-2.5 flex items-center justify-between gap-2 z-10">
          <div className="flex items-center gap-2 min-w-0">
            {devices.length > 1 ? (
              <select
                value={selectedDeviceId}
                onChange={(e) => handleSwitchDevice(e.target.value)}
                aria-label="Select camera device"
                className="min-h-[40px] max-w-[190px] px-2.5 rounded-xl bg-zinc-900/85 text-zinc-100 border border-zinc-700 text-xs font-medium focus:outline-none truncate"
              >
                {devices.map((dev, idx) => (
                  <option key={dev.deviceId || idx} value={dev.deviceId}>
                    {dev.label || `Camera ${idx + 1}`}
                  </option>
                ))}
              </select>
            ) : (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-zinc-950/75 text-zinc-200 text-[11px] font-medium backdrop-blur-sm">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    liveReady ? 'bg-emerald-400' : 'bg-zinc-500'
                  }`}
                />
                <span>{statusNote || 'Auto-detects printed & handwriting'}</span>
              </span>
            )}
          </div>

          {!embedded && onClose && (
            <button
              type="button"
              onClick={() => {
                stopStream();
                onClose();
              }}
              className="min-h-[44px] min-w-[44px] flex items-center justify-center rounded-xl bg-zinc-900/85 text-zinc-200 hover:text-white border border-zinc-700 shrink-0"
              aria-label="Close camera"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Hidden inputs: Gallery upload (all image files) & Native Device Camera fallback */}
      <input
        ref={galleryInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileInputChange}
        className="hidden"
      />
      <input
        ref={nativeInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleFileInputChange}
        className="hidden"
      />

      {/* Thumb-Zone Primary Action Bar (>=44px / 48px touch targets, fits cleanly at 390px) */}
      <div className="flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={handleFlipCamera}
          disabled={paused}
          className="min-h-[44px] px-3 sm:px-3.5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-800 disabled:opacity-50 inline-flex items-center justify-center gap-1.5 whitespace-nowrap shrink-0"
          aria-label="Flip camera"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Flip</span>
        </button>

        <button
          type="button"
          onClick={handleCaptureFrame}
          disabled={!liveReady || isCapturing}
          className="min-h-[48px] flex-1 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-700 active:scale-[0.98] disabled:opacity-50 text-white text-sm font-medium inline-flex items-center justify-center gap-2 whitespace-nowrap transition-all"
        >
          <Camera className="w-4 h-4 shrink-0" />
          <span>Take Photo</span>
        </button>

        <button
          type="button"
          onClick={() => galleryInputRef.current?.click()}
          className="min-h-[44px] px-3 sm:px-3.5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-800 inline-flex items-center justify-center gap-1.5 whitespace-nowrap shrink-0"
          aria-label="Choose photo from gallery"
        >
          <ImageIcon className="w-3.5 h-3.5" />
          <span>Gallery</span>
        </button>
      </div>

      {/* Subtle Secondary Row: Native OS Device Cam + Paste Hint */}
      <div className="flex items-center justify-between gap-2 px-0.5 text-[11px] text-zinc-500 dark:text-zinc-400">
        <button
          type="button"
          onClick={() => nativeInputRef.current?.click()}
          className="min-h-[36px] inline-flex items-center gap-1 hover:text-zinc-900 dark:hover:text-zinc-200 underline underline-offset-2"
        >
          <span>Use native device camera</span>
        </button>
        <span className="truncate">Drop image or paste (Ctrl+V)</span>
      </div>
    </div>
  );
};

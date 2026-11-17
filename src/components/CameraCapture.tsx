import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Camera, RefreshCw, X, Upload, Loader2 } from 'lucide-react';

interface CameraCaptureProps {
  onCapture: (dataUrl: string) => void;
  onClose: () => void;
}

export const CameraCapture: React.FC<CameraCaptureProps> = ({ onCapture, onClose }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nativeInputRef = useRef<HTMLInputElement | null>(null);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');
  const [isStarting, setIsStarting] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // ignore track stop failure
        }
      });
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
      const videoInputs = all.filter((d) => d.kind === 'videoinput');
      setDevices(videoInputs);
    } catch {
      // ignore enumeration errors
    }
  }, []);

  const startCamera = useCallback(
    async (opts?: { deviceId?: string; facing?: 'environment' | 'user' }) => {
      stopStream();
      setIsStarting(true);
      setErrorMsg(null);

      if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
        setIsStarting(false);
        setErrorMsg(
          'Live camera streaming is unavailable in this browser context. Use the device camera button below to take a photo directly.'
        );
        return;
      }

      const targetDeviceId = opts?.deviceId ?? selectedDeviceId;
      const targetFacing = opts?.facing ?? facingMode;

      // Progressive constraint fallback chain so any webcam, laptop, tablet, or phone camera works
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
        try {
          acquiredStream = await navigator.mediaDevices.getUserMedia(constraints);
          break;
        } catch (err: any) {
          lastErr = err;
          // If permission was explicitly denied or hardware is in use, don't keep retrying other constraints
          if (
            err?.name === 'NotAllowedError' ||
            err?.name === 'PermissionDeniedError' ||
            err?.name === 'NotReadableError'
          ) {
            break;
          }
        }
      }

      if (!acquiredStream) {
        setIsStarting(false);
        const name = lastErr?.name || '';
        if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
          setErrorMsg(
            'Camera permission was denied. Allow camera access in your browser settings, or capture with your device camera below.'
          );
        } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
          setErrorMsg(
            'No live video camera was detected on this device. You can still upload or capture a photo below.'
          );
        } else if (name === 'NotReadableError') {
          setErrorMsg(
            'Your camera appears to be in use by another app. Close the other app and retry, or use device capture below.'
          );
        } else {
          setErrorMsg(
            lastErr?.message ||
              'Could not start live camera preview. Use the device camera button below instead.'
          );
        }
        return;
      }

      streamRef.current = acquiredStream;

      const activeTrack = acquiredStream.getVideoTracks()[0];
      const activeSettings = activeTrack?.getSettings?.();
      if (activeSettings?.deviceId && !targetDeviceId) {
        setSelectedDeviceId(activeSettings.deviceId);
      }

      await refreshDevices();

      if (videoRef.current) {
        videoRef.current.srcObject = acquiredStream;
        try {
          await videoRef.current.play();
        } catch {
          // autoPlay + playsInline will still display stream on browsers that restrict programmatic play
        }
      }

      setIsStarting(false);
    },
    [facingMode, refreshDevices, selectedDeviceId, stopStream]
  );

  useEffect(() => {
    startCamera();
    return () => {
      stopStream();
    };
    // Run initial mount start only; subsequent device/facing changes invoke startCamera directly
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSwitchDevice = (nextDeviceId: string) => {
    setSelectedDeviceId(nextDeviceId);
    startCamera({ deviceId: nextDeviceId });
  };

  const handleFlipCamera = () => {
    const nextFacing = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(nextFacing);
    setSelectedDeviceId('');
    startCamera({ deviceId: '', facing: nextFacing });
  };

  const handleCaptureFrame = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
    stopStream();
    onCapture(dataUrl);
  };

  const handleNativeFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
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

  return (
    <div className="space-y-3">
      <div className="relative rounded-2xl overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-zinc-950 aspect-[3/4] sm:aspect-[4/3] max-h-[56vh] w-full flex items-center justify-center">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover ${isStarting || errorMsg ? 'opacity-0' : 'opacity-100'}`}
        />

        {isStarting && !errorMsg && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-zinc-300 text-xs">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span>Starting camera...</span>
          </div>
        )}

        {errorMsg && (
          <div className="absolute inset-0 p-5 flex flex-col items-center justify-center text-center gap-3 bg-zinc-950/95">
            <p className="text-xs text-rose-300 max-w-xs leading-relaxed">{errorMsg}</p>
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
                onClick={() => nativeInputRef.current?.click()}
                className="min-h-[44px] px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium inline-flex items-center gap-1.5 whitespace-nowrap"
              >
                <Upload className="w-3.5 h-3.5" />
                <span>Open Device Camera</span>
              </button>
            </div>
          </div>
        )}

        {/* Top overlay controls: device picker & close */}
        <div className="absolute top-2.5 left-2.5 right-2.5 flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 min-w-0">
            {devices.length > 1 && (
              <select
                value={selectedDeviceId}
                onChange={(e) => handleSwitchDevice(e.target.value)}
                aria-label="Select camera device"
                className="min-h-[40px] max-w-[200px] px-2.5 rounded-xl bg-zinc-900/85 text-zinc-100 border border-zinc-700 text-xs font-medium focus:outline-none truncate"
              >
                {devices.map((dev, idx) => (
                  <option key={dev.deviceId || idx} value={dev.deviceId}>
                    {dev.label || `Camera ${idx + 1}`}
                  </option>
                ))}
              </select>
            )}
          </div>

          <button
            type="button"
            onClick={() => {
              stopStream();
              onClose();
            }}
            className="min-h-[44px] min-w-[44px] flex items-center justify-center rounded-xl bg-zinc-900/85 text-zinc-200 hover:text-white border border-zinc-700"
            aria-label="Close camera"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Hidden native camera input fallback for any OS/device */}
      <input
        ref={nativeInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleNativeFileChange}
        className="hidden"
      />

      {/* Thumb-zone Camera action bar */}
      <div className="flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={handleFlipCamera}
          className="min-h-[44px] px-3.5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 inline-flex items-center gap-1.5 whitespace-nowrap shrink-0"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Flip</span>
        </button>

        <div className="flex items-center gap-2 flex-1 justify-end">
          <button
            type="button"
            onClick={() => nativeInputRef.current?.click()}
            className="min-h-[44px] px-3.5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-950 text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 whitespace-nowrap shrink-0"
          >
            Native Cam
          </button>
          <button
            type="button"
            onClick={handleCaptureFrame}
            disabled={isStarting || Boolean(errorMsg)}
            className="min-h-[48px] flex-1 sm:flex-initial px-5 rounded-xl bg-indigo-600 hover:bg-indigo-700 active:scale-[0.98] disabled:opacity-50 text-white text-sm font-medium inline-flex items-center justify-center gap-2 whitespace-nowrap shrink-0 transition-transform"
          >
            <Camera className="w-4 h-4" />
            <span>Take Photo</span>
          </button>
        </div>
      </div>
    </div>
  );
};
